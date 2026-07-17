/**
 * @file ManusZmqBridge.cpp
 * @brief Standalone headless MANUS → ZMQ bridge.
 *
 * Self-contained binary that initializes the MANUS SDK in Core Integrated
 * mode, registers its own callbacks, and publishes skeleton data as JSON
 * over a ZMQ PUB socket. Does NOT depend on or modify SDKClient.
 *
 * This software contains source code provided by Manus Technology Group B.V.
 *
 * Usage:
 *     ./ManusZmqBridge.out
 */

#include "ManusSDK.h"
#include "ManusSDKTypes.h"
#include "ZmqPublisher.hpp"

#include <csignal>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
static volatile bool g_Running = true;
static ZmqPublisher g_Publisher("tcp://*:8000");

// Landscape data (glove ID → side mapping)
static std::mutex g_LandscapeMutex;
static uint32_t g_LeftGloveID = 0;
static uint32_t g_RightGloveID = 0;

// Frame counter for debug logging
static std::atomic<uint64_t> g_FrameCount{0};

// ---------------------------------------------------------------------------
// Signal handler
// ---------------------------------------------------------------------------
static void SignalHandler(int)
{
    g_Running = false;
}

// ---------------------------------------------------------------------------
// SDK Callbacks
// ---------------------------------------------------------------------------

static void OnConnectedCallback(const ManusHost* const)
{
    printf("[Bridge] Connected to MANUS Core.\n");
    fflush(stdout);

    // Enable raw skeleton streaming.
    SDKReturnCode result = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
    if (result != SDKReturnCode_Success)
        printf("[Bridge] WARNING: Failed to set hand motion mode: %d\n", (int)result);
    else
        printf("[Bridge] Raw skeleton stream enabled (HandMotion_Auto).\n");
    fflush(stdout);
}

static void OnDisconnectedCallback(const ManusHost* const)
{
    printf("[Bridge] Disconnected from MANUS Core.\n");
    fflush(stdout);
}

static void OnLandscapeCallback(const Landscape* const p_Landscape)
{
    std::lock_guard<std::mutex> lock(g_LandscapeMutex);
    g_LeftGloveID = 0;
    g_RightGloveID = 0;

    for (uint32_t i = 0; i < p_Landscape->gloveDevices.gloveCount; i++)
    {
        auto& glove = p_Landscape->gloveDevices.gloves[i];
        if (g_LeftGloveID == 0 && glove.side == Side_Left)
            g_LeftGloveID = glove.id;
        else if (g_RightGloveID == 0 && glove.side == Side_Right)
            g_RightGloveID = glove.id;
    }

    static uint32_t s_PrintCount = 0;
    if (s_PrintCount < 5 || s_PrintCount % 100 == 0)
    {
        printf("[Bridge] Landscape: %u gloves | L=%u R=%u\n",
               p_Landscape->gloveDevices.gloveCount, g_LeftGloveID, g_RightGloveID);
        fflush(stdout);
    }
    s_PrintCount++;
}

static void OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_Info)
{
    for (uint32_t i = 0; i < p_Info->skeletonsCount; i++)
    {
        RawSkeletonInfo info;
        CoreSdk_GetRawSkeletonInfo(i, &info);

        std::vector<SkeletonNode> nodes(info.nodesCount);
        CoreSdk_GetRawSkeletonData(i, nodes.data(), info.nodesCount);

        // Determine side from glove ID
        std::string side = "UNKNOWN";
        {
            std::lock_guard<std::mutex> lock(g_LandscapeMutex);
            if (g_RightGloveID != 0 && info.gloveId == g_RightGloveID)
                side = "RIGHT";
            else if (g_LeftGloveID != 0 && info.gloveId == g_LeftGloveID)
                side = "LEFT";
        }

        g_Publisher.Publish(side, info.gloveId, nodes.data(), info.nodesCount);

        uint64_t count = ++g_FrameCount;
        if (count <= 3 || count % 500 == 0)
        {
            printf("[Bridge] Frame #%lu | side=%s glove=%u nodes=%u\n",
                   count, side.c_str(), info.gloveId, info.nodesCount);
            fflush(stdout);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);

    printf("=== MANUS ZMQ Bridge (standalone) ===\n");

    // Initialize SDK in Core Integrated mode (no separate MANUS Core daemon).
    printf("[Bridge] Initializing SDK (Core Integrated)...\n");
    SDKReturnCode result = CoreSdk_InitializeIntegrated();
    if (result != SDKReturnCode_Success)
    {
        printf("[Bridge] ERROR: CoreSdk_InitializeIntegrated failed: %d\n", (int)result);
        return 1;
    }

    // Register callbacks
    CoreSdk_RegisterCallbackForOnConnect(OnConnectedCallback);
    CoreSdk_RegisterCallbackForOnDisconnect(OnDisconnectedCallback);
    CoreSdk_RegisterCallbackForLandscapeStream(OnLandscapeCallback);
    CoreSdk_RegisterCallbackForRawSkeletonStream(OnRawSkeletonStreamCallback);

    // Set coordinate system: Z-up, X-from-viewer, right-handed, meters.
    CoordinateSystemVUH vuh;
    CoordinateSystemVUH_Init(&vuh);
    vuh.handedness = Side_Right;
    vuh.up = AxisPolarity_PositiveZ;
    vuh.view = AxisView_XFromViewer;
    vuh.unitScale = 1.0f;
    result = CoreSdk_InitializeCoordinateSystemWithVUH(vuh, true);
    if (result != SDKReturnCode_Success)
    {
        printf("[Bridge] ERROR: CoreSdk_InitializeCoordinateSystemWithVUH failed: %d\n", (int)result);
        CoreSdk_ShutDown();
        return 1;
    }

    // Connect (integrated mode uses an empty host).
    printf("[Bridge] Connecting...\n");
    ManusHost emptyHost;
    ManusHost_Init(&emptyHost);

    while (g_Running)
    {
        result = CoreSdk_ConnectToHost(emptyHost);
        if (result == SDKReturnCode_Success)
            break;
        printf("[Bridge] Not connected yet, retrying in 1s...\n");
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if (!g_Running)
    {
        CoreSdk_ShutDown();
        return 0;
    }

    printf("[Bridge] Running. Publishing on tcp://*:8000. Ctrl+C to stop.\n");
    fflush(stdout);

    // Main loop — just keep the process alive. Callbacks do all the work.
    while (g_Running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    printf("\n[Bridge] Shutting down...\n");
    CoreSdk_ShutDown();
    printf("[Bridge] Done.\n");

    return 0;
}
