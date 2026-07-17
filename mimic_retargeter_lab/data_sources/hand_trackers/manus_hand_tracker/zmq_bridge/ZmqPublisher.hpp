/**
 * @file ZmqPublisher.hpp
 * @brief ZMQ PUB publisher for streaming MANUS skeleton data as JSON.
 *
 * Publishes on ZMQ PUB socket with topic-prefixed messages:
 *   "RIGHT {json...}"  or  "LEFT {json...}"
 *
 * Subscribers use zmq.SUB with topic filter "RIGHT" or "LEFT" to receive
 * only the hand they care about.
 *
 * JSON payload per message:
 *   {
 *     "glove_id": 12345,
 *     "side": "RIGHT",
 *     "nodes": [
 *       {"id": 0, "pos": [x, y, z], "quat": [w, x, y, z]},
 *       ...
 *     ]
 *   }
 */

#pragma once

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <cstdio>
#include <atomic>

class ZmqPublisher
{
public:
    ZmqPublisher(const std::string& endpoint = "tcp://*:8000")
        : m_Context(1)
        , m_Socket(m_Context, zmq::socket_type::pub)
    {
        m_Socket.set(zmq::sockopt::sndhwm, 1);
        m_Socket.bind(endpoint);
        printf("[ZMQ] PUB bound to %s\n", endpoint.c_str());
    }

    ~ZmqPublisher()
    {
        m_Socket.close();
        m_Context.close();
    }

    /// Publish one skeleton frame as a topic-prefixed JSON message.
    /// side: "LEFT" or "RIGHT"
    template<typename NodeT>
    void Publish(const std::string& side, uint32_t gloveId, const NodeT* nodes, uint32_t nodeCount)
    {
        nlohmann::json j;
        j["glove_id"] = gloveId;
        j["side"] = side;

        auto& jNodes = j["nodes"];
        jNodes = nlohmann::json::array();

        for (uint32_t i = 0; i < nodeCount; i++)
        {
            const auto& t = nodes[i].transform;
            nlohmann::json node;
            node["id"] = nodes[i].id;
            node["pos"] = {t.position.x, t.position.y, t.position.z};
            // Quaternion in MANUS SDK order: w, x, y, z
            node["quat"] = {t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z};
            jNodes.push_back(std::move(node));
        }

        // ZMQ PUB topic filtering matches on message prefix.
        // Format: "RIGHT {json...}" — subscriber with filter "RIGHT" receives this.
        std::string msg = side + " " + j.dump();
        m_Socket.send(zmq::buffer(msg), zmq::send_flags::dontwait);

        m_FrameCount++;
        if (m_FrameCount % 100 == 1)
        {
            printf("[ZMQ] Published frame #%lu | side=%s glove=%u nodes=%u\n",
                   m_FrameCount.load(), side.c_str(), gloveId, nodeCount);
        }
    }

    unsigned long GetFrameCount() const { return m_FrameCount.load(); }

private:
    zmq::context_t m_Context;
    zmq::socket_t  m_Socket;
    std::atomic<unsigned long> m_FrameCount{0};
};
