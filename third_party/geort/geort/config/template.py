# Welcome to the hand setup guide!

allegro_hand_config = {
    "name": "allegro_right",  # Todo: name your robot hand.
    # URDF path.
    # Todo: Put your urdf under the assets folder, and fill in this line.
    "urdf_path": "./assets/allegro_right/allegro_hand_right.urdf",
    # The name of baselink in urdf.
    # In GeoRT, we represent keypoint position in the base_link frame.
    # Todo:
    # 1. Fill in the hand base_link name specified by your urdf.
    # 2. Please refer to the doc (or below) for the base link orientation convention.
    #    Y: center of palm to the thumb. Z: center of palm to the middle finger. X: normal of palm.
    # 3. If your base link uses a different orientation:
    #    you can add a virtual base_link and joint to your urdf, and change this to the new virtual base_link.
    "base_link": "base_link",
    # To be completed by user.
    # Important: The GeoRT will return joint position in this order!
    # Todo:
    # 1. Fill in the hand joint names specified by your URDF.
    # Recommendation: For convenience, specify this according to your robot controller API format.
    "joint_order": [
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ],
    # To be completed by user.
    # Fingertip links in your urdf. This is a list.
    "fingertip_link": [
        # Now we are going to add a bunch of fingertip information as below.
        {
            # name annotation. you can use whatever you like.
            "name": "index",
            # the corresponding link in your URDF.
            "link": "link_3.0_tip",
            # all the joints in URDF that drive the link (i.e. link_3.0_tip)
            # (apologies, this should have been automatically extracted from URDF but this also gives you some control over data format...)
            "joint": ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0"],
            # The center of robot fingertip viewed from the link frame (i.e. link_3.0_tip).
            "center_offset": [0.0, 0.0, -0.005],
            # The corresponding fingertip id in your [N, 3] human hand mocap keypoint representation.
            # For instance, the index finger id is 8 in MediaPipe keypoint representation (N=21).
            "human_hand_id": 8,
        },
        # Index finger Done!
        # Now we move on to middle, ring, ...
        {
            "name": "middle",
            "link": "link_7.0_tip",
            "joint": ["joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 12,
        },
        {
            "name": "ring",
            "link": "link_11.0_tip",
            "joint": ["joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 16,
        },
        {
            "name": "thumb",
            "link": "link_15.0_tip",
            "joint": ["joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0"],
            "center_offset": [0.0, 0.0, -0.005],
            "human_hand_id": 4,
        },
    ],
    # all set! we are ready to go!
}
