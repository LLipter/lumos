//
// Created by 了然 on 2018/12/25.
//

#ifndef FFMPEG_PACKET_QUEUE_H
#define FFMPEG_PACKET_QUEUE_H

#include <libavcodec/avcodec.h>

typedef struct Packet_Queue{
    int full_size;
    int cur_size;
    int head_pos;
    AVPacket **packet_array;
}Packet_Queue;

void packet_queue_alloc(Packet_Queue *packet_queue, int size);

void packet_queue_free(Packet_Queue *packet_queue);

int push_packet(Packet_Queue *packet_queue, AVPacket *packet);

AVPacket *pop_packet(Packet_Queue *packet_queue);

#endif //FFMPEG_PACKET_QUEUE_H
