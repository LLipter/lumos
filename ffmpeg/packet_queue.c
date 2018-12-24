//
// Created by 了然 on 2018/12/25.
//

#include "packet_queue.h"

int full_size;
int cur_size;
int head_pos;
AVPacket **packet_array;

void packet_queue_alloc(int size) {
    full_size = size;
    cur_size = 0;
    head_pos = 0;
    packet_array = malloc(sizeof(AVPacket *) * size);
}

void packet_queue_free(){
    free(packet_array);
}

int push_packet(AVPacket *packet){
    if(cur_size == full_size)
        return -1;
    int pos = (head_pos + cur_size) % full_size;
    packet_array[pos] = packet;
    cur_size++;
    return 0;
}

AVPacket * pop_packet(){
    if(cur_size == 0)
        return NULL;
    AVPacket * packet = packet_array[head_pos];
    head_pos = (head_pos + 1) % full_size;
    cur_size--;
    return packet;
}

