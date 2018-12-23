#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#include "io.h"

#define BUFF_SIZE 1024

char *cmd = "split";
char *input_dir = "../video";
char *filename = "02.mp4";
char save_path[BUFF_SIZE];
char input_path[BUFF_SIZE];
char buf[BUFF_SIZE];
int width = 0;
int height = 0;
int ret = 0;
int video_stream_index = -1;
AVFormatContext *pFormatCtx = NULL;
AVCodec *pCodec = NULL;
AVCodecContext *codec_ctx = NULL;
AVCodecParameters *pCodePara = NULL;
AVPacket *packet = NULL;
AVFrame *frame = NULL;


void cleanup(char *msg) {
    if (pFormatCtx)
        avformat_close_input(&pFormatCtx);
    if (codec_ctx)
        avcodec_free_context(&codec_ctx);
    if (packet)
        av_packet_unref(packet);
    if (frame)
        av_frame_free(&frame);
    if (msg) {
        perror(msg);
        exit(1);
    }
}

void init() {
    if (snprintf(input_path, BUFF_SIZE, "%s/%s", input_dir, filename) < 0)
        cleanup("Error in snprintf");
    if (snprintf(save_path, BUFF_SIZE, "%s-IFrame", input_path) < 0)
        cleanup("Error in snprintf");
}

void save_frame_as_jpeg(AVFrame *pFrame, char *filename) {
    AVCodec *jpegCodec = avcodec_find_encoder(AV_CODEC_ID_JPEG2000);
    if (!jpegCodec)
        cleanup("Error in finding jpeg decoder");
    AVCodecContext *jpegContext = avcodec_alloc_context3(jpegCodec);
    if (!jpegContext)
        cleanup("Error in create jpeg context");
    jpegContext->pix_fmt = codec_ctx->pix_fmt;
    jpegContext->height = height;
    jpegContext->width = width;
    jpegContext->time_base = codec_ctx->time_base;
    if (avcodec_open2(jpegContext, jpegCodec, NULL) < 0)
        cleanup("Error in opening jpeg decoder");
    AVPacket *jpeg_packet = av_packet_alloc();

    if (avcodec_send_frame(jpegContext, pFrame) < 0)
        cleanup("Error in sending frame");
    if (avcodec_send_frame(jpegContext, NULL) < 0)
        cleanup("Error in sending frame");
    if (avcodec_receive_packet(jpegContext, jpeg_packet) < 0)
        cleanup("Error in receiving packet");

    FILE *JPEGFile = fopen(filename, "wb");
    fwrite(jpeg_packet->data, 1, jpeg_packet->size, JPEGFile);
    fclose(JPEGFile);

    avcodec_close(jpegContext);
    av_packet_unref(jpeg_packet);
}

void load_jpeg_as_frame(AVFrame *pFrame, char *filename) {
//    AVCodec *jpegCodec = avcodec_find_encoder(AV_CODEC_ID_JPEG2000);
//    if (!jpegCodec)
//        cleanup("Error in finding jpeg decoder");
//    AVCodecContext *jpegContext = avcodec_alloc_context3(jpegCodec);
//    if (!jpegContext)
//        cleanup("Error in create jpeg context");
//    jpegContext->pix_fmt = codec_ctx->pix_fmt;
//    jpegContext->height = height;
//    jpegContext->width = width;
//    jpegContext->time_base = codec_ctx->time_base;
//    if (avcodec_open2(jpegContext, jpegCodec, NULL) < 0)
//        cleanup("Error in opening jpeg decoder");
//
//
//    int size = get_file_size(filename);
//    AVPacket *jpeg_packet = av_packet_alloc();
//    if (av_new_packet(jpeg_packet, size) < 0)
//        cleanup("Error in allocating packet");
//
//    FILE *JPEGFile = fopen(filename, "rb");
//    fread(jpeg_packet->data, 1, jpeg_packet->size, JPEGFile);
//    fclose(JPEGFile);
//    printf("!11");
//    if (avcodec_send_packet(jpegContext, jpeg_packet) < 0)
//        cleanup("Error in sending packet");
//    printf("!12");
//    if (avcodec_send_packet(jpegContext, NULL) < 0)
//        cleanup("Error in sending packet");
//    printf("!13");
//    if (avcodec_receive_packet(codec_ctx, pFrame) < 0)
//        cleanup("Error in receiving frame");
//
//    av_packet_unref(jpeg_packet);
}

void extract_frame(AVFrame *frame) {
    if (frame->pict_type == AV_PICTURE_TYPE_I) {
        snprintf(buf, BUFF_SIZE, "%s/%s-%d.jpg", save_path, filename, codec_ctx->frame_number);
        if (strcmp(cmd, "split") == 0)
            save_frame_as_jpeg(frame, buf);
        else
            load_jpeg_as_frame(frame, buf);
    }

    if (strcmp(cmd, "split") != 0) {

    }
}


int main(int argc, char **argv) {
    if (argc <= 3) {
        fprintf(stderr, "Usage: %s <command> <dirpath> <filename>\n", argv[0]);
        exit(0);
    }
    cmd = argv[1];
    input_dir = argv[2];
    filename = argv[3];

    init();

    if (strcmp(cmd, "split") == 0) {
        // remove old data
        remove_directory(save_path);
        if (mkdir(save_path, 0777) < 0)
            cleanup("Error in mkdir");
    }

    //2、打开视频文件
    pFormatCtx = avformat_alloc_context();
    if ((avformat_open_input(&pFormatCtx, input_path, NULL, NULL)) < 0)
        cleanup("Error in opening input file");

    //3、获取视频信息
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
        cleanup("Error in finding stream");

    //4、找到视频流的位置
    video_stream_index = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);
    if (video_stream_index == AVERROR_STREAM_NOT_FOUND)
        cleanup("Error in finding stream index");
    else if (video_stream_index == AVERROR_DECODER_NOT_FOUND)
        cleanup("Error in finding decoder");

    //5、获取AVCodecParameters
    pCodePara = pFormatCtx->streams[video_stream_index]->codecpar;
    width = pCodePara->width;
    height = pCodePara->height;

    //6、打开解码器
    codec_ctx = avcodec_alloc_context3(pCodec);
    avcodec_parameters_to_context(codec_ctx, pCodePara);
    if (avcodec_open2(codec_ctx, pCodec, NULL) < 0)
        cleanup("Error in opening codec");

    //7、解析每一帧数据
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    // read all frames and send them into decoder
    int cnt = 1;
    while (av_read_frame(pFormatCtx, packet) >= 0) {
        printf("%d\n", cnt++);
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0)
                cleanup("Error in sending a packet for decoding");
            while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0)
                extract_frame(frame);
            if (ret == AVERROR(EAGAIN))
                continue;
            else if (ret < 0)
                cleanup("Error in receiving a packet from decoder");
        }
    }
    // send flush packet, enter draining mode.
    avcodec_send_packet(codec_ctx, NULL);
    while ((ret = avcodec_receive_frame(codec_ctx, frame)) >= 0)
        extract_frame(frame);
    if (ret != AVERROR_EOF)
        cleanup("Error in draining decoder stream");

    cleanup(NULL);
}


