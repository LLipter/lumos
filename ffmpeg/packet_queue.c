//
// Created by 了然 on 2018/12/25.
//

#include "packet_queue.h"


void packet_queue_alloc(Packet_Queue *packet_queue, int size) {
    packet_queue->full_size = size;
    packet_queue->cur_size = 0;
    packet_queue->head_pos = 0;
    packet_queue->packet_array = malloc(sizeof(AVPacket *) * size);
}

void packet_queue_free(Packet_Queue *packet_queue) {
    free(packet_queue->packet_array);
}

int push_packet(Packet_Queue *packet_queue, AVPacket *packet) {
    if (packet_queue->cur_size == packet_queue->full_size)
        return -1;
    int pos = (packet_queue->head_pos + packet_queue->cur_size) % packet_queue->full_size;
    packet_queue->packet_array[pos] = packet;
    packet_queue->cur_size++;
    return 0;
}

AVPacket *pop_packet(Packet_Queue *packet_queue) {
    if (packet_queue->cur_size == 0)
        return NULL;
    AVPacket *packet = packet_queue->packet_array[packet_queue->head_pos];
    packet_queue->head_pos = (packet_queue->head_pos + 1) % packet_queue->full_size;
    packet_queue->cur_size--;
    return packet;
}

