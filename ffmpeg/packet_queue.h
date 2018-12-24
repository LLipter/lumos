//
// Created by 了然 on 2018/12/25.
//

#ifndef FFMPEG_PACKET_QUEUE_H
#define FFMPEG_PACKET_QUEUE_H

#include <libavcodec/avcodec.h>

void queue_alloc(int size);
void queue_free();
int push(AVPacket * packet);
AVPacket * pop();

#endif //FFMPEG_PACKET_QUEUE_H
