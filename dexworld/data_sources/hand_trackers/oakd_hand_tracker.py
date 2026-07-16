import logging
from multiprocessing import Manager, Process

import numpy as np
import omegaconf

from dexworld.data_sources.hand_trackers.depthai_hand_tracker import (
    HandTracker,
    HandTrackerEdge,
    HandTrackerRenderer,
)
from dexworld.data_sources.hand_trackers.depthai_hand_tracker.Filters import (
    LandmarksSmoothingFilter,
)
from dexworld.hand_models import ManoKeypointHandModel


def _run_tracker(use_edge, tracker_cfg, out_data, start_event, logger):
    if use_edge:
        tracker = HandTrackerEdge(**tracker_cfg, logger=logger)
    else:
        tracker = HandTracker(**tracker_cfg, logger=logger)

    smoothing_filter = LandmarksSmoothingFilter(
        min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True
    )

    logger.info("Starting tracker loop... Press 'q' or 'ESC' to quit.")

    while True:
        # Run hand tracker on next frame
        frame, hands, bag = tracker.next_frame()
        if len(hands) == 1:
            out_data["hands"] = smoothing_filter.apply(
                hands[0].world_landmarks, object_scale=hands[0].rect_w_a
            )
            out_data["hand_2d"] = hands[0].landmarks
        else:
            out_data["hands"] = None
            out_data["hand_2d"] = None

        out_data["frame"] = frame
        out_data["bag"] = bag
        if not start_event.is_set():
            start_event.set()
        if frame is None:
            logger.info("End of video stream.")
            break


class OakDHandTracker:
    """
    A wrapper class that initializes and runs the HandTracker
    and HandTrackerRenderer based on a Hydra config object.
    """

    def __init__(self, cfg: omegaconf.DictConfig, logger=logging.getLogger(__name__)):
        """
        Initializes the tracker and renderer.

        Args:
            cfg: The Hydra DictConfig object.
        """
        self.cfg = cfg
        self.hand_model = ManoKeypointHandModel()
        self.manager = Manager()
        self.out_data = self.manager.dict()
        self.start_event = self.manager.Event()
        self.logger = logger

        # Convert the 'tracker' part of the config to a standard python dict.
        # This resolves any interpolations and makes it easy to pass as **kwargs.
        self.tracker_config = omegaconf.OmegaConf.to_container(
            cfg.tracker, resolve=True
        )

        # if cfg.use_edge:
        #     # HandTrackerEdge accepts the 'use_same_image' argument
        #     print("Initializing in Edge mode...")
        #     self.tracker = HandTrackerEdge(**tracker_config)
        # else:
        #     # HandTracker (host) does not accept 'use_same_image', so we remove it
        #     # to avoid an 'unexpected keyword argument' error.
        #     print("Initializing in Host mode...")
        #     tracker_config.pop("use_same_image", None)
        #     self.tracker = HandTracker(**tracker_config)

        # print("Initialization complete.")
        # # Initialize the renderer
        # self.renderer = HandTrackerRenderer(
        #     tracker=self.tracker, output=cfg.renderer.output
        # )

    def _start_tracker(self):
        self.tracker_process = Process(
            target=_run_tracker,
            args=(
                self.cfg.use_edge,
                self.tracker_config,
                self.out_data,
                self.start_event,
                self.logger,
            ),
        )
        self.tracker_process.start()

    def run_blocking_viz(self):
        """
        Starts the main processing loop.
        """
        print("Starting tracker loop... Press 'q' or 'ESC' to quit.")
        if self.cfg.use_edge:
            # HandTrackerEdge accepts the 'use_same_image' argument
            print("Initializing in Edge mode...")
            self.tracker = HandTrackerEdge(**self.tracker_config)
        else:
            # HandTracker (host) does not accept 'use_same_image', so we remove it
            # to avoid an 'unexpected keyword argument' error.
            print("Initializing in Host mode...")
            self.tracker_config.pop("use_same_image", None)
            self.tracker = HandTracker(**self.tracker_config)

        # Initialize the renderer
        self.renderer = HandTrackerRenderer(
            tracker=self.tracker, output=self.cfg.renderer.output
        )
        print("Initialization complete.")
        try:
            while True:
                # Run hand tracker on next frame
                frame, hands, bag = self.tracker.next_frame()
                if frame is None:
                    print("End of video stream.")
                    break

                # Draw hands
                frame = self.renderer.draw(frame, hands, bag)

                # Show frame
                key = self.renderer.waitKey(delay=1)
                if key == 27 or key == ord("q"):
                    print("Quit key pressed.")
                    break
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt. Exiting...")
        finally:
            self.close()

    def run_non_blocking(self):
        self._start_tracker()
        self.logger.info("Waiting for tracker to start...")
        self.start_event.wait()
        try:
            while True:
                frame, hands, _hands_2d, _bag = (
                    self.out_data["frame"],
                    self.out_data["hands"],
                    self.out_data["hand_2d"],
                    self.out_data["bag"],
                )

                if frame is None:
                    print("End of video stream.")
                    break

                # if hands_2d is not None:
                #     for keypoint_2d in hands_2d:
                #         cv2.circle(
                #             frame,
                #             (int(keypoint_2d[0]), int(keypoint_2d[1])),
                #             5,
                #             (0, 255, 0),
                #             -1,
                #         )

                #     for pair in LINES_HAND:
                #         cv2.line(
                #             frame,
                #             (int(hands_2d[pair[0]][0]), int(hands_2d[pair[0]][1])),
                #             (int(hands_2d[pair[1]][0]), int(hands_2d[pair[1]][1])),
                #             (0, 255, 0),
                #             2,
                #         )
                # cv2.imshow("Hand Tracker", frame)
                # cv2.waitKey(1)

                yield hands

        except KeyboardInterrupt:
            self.logger.info("\nCaught KeyboardInterrupt. Exiting...")
            self.tracker_process.terminate()
        finally:
            self.close()

    def get_iter(self):
        for hands in self.run_non_blocking():
            if hands is None:
                continue

            hands = np.asarray(hands, dtype=np.float32)
            ret = {
                "transforms": np.stack(
                    [
                        t
                        for t_dict in self.hand_model.from_joints(
                            hands[np.newaxis]
                        ).to_raw_frames()
                        for t in t_dict.values()
                    ]
                ),
                "joints": hands[np.newaxis],
                "links": [],
            }
            yield ret

    def close(self):
        """
        Cleans up and exits the tracker and renderer.
        """
        print("Exiting tracker and renderer...")
        if hasattr(self, "renderer"):
            self.renderer.exit()
        if hasattr(self, "tracker"):
            self.tracker.exit()
        print("Cleanup finished.")
