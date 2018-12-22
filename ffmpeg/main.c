#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

char *input_path = "../video/01.mp4";
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

int main(int argc, char **argv) {
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
    int got_picture_ptr, frame_count = 1;
    packet = av_packet_alloc();
    frame = av_frame_alloc();

    // read all frames and send them into decoder
    int cnt = 1;
    while (av_read_frame(pFormatCtx, packet) >= 0) {
        printf("%d\n", cnt++);
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0)
                cleanup("Error in sending a packet for decoding");
            while (ret = avcodec_receive_frame(codec_ctx, frame) > 0) {
                // do something
            }
            if (ret == AVERROR(EAGAIN))
                continue;
            else if (ret < 0)
                cleanup("Error in receiving a packet from decoder");
        }
    }
    // send flush packet, enter draining mode.
    avcodec_send_packet(NULL, NULL);
    while (ret = avcodec_receive_frame(codec_ctx, frame) > 0) {
        // do something
    }
    if (ret != AVERROR_EOF)
        cleanup("Error in draining decoder stream");
    return 0;

    //一帧一帧读取压缩的视频数据
    while (av_read_frame(pFormatCtx, packet) >= 0) {
        //找到视频流
        if (packet->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, packet) < 0)
                cleanup("Cannot sending a packet for decoding");
            ret = avcodec_receive_frame(codec_ctx, frame);
            if (ret == AVERROR(EINVAL))
                cleanup("0 receiving a frame from decoder");
            if (ret == AVERROR(EAGAIN))
                cleanup("1 receiving a frame from decoder");
            else if (ret == AVERROR_EOF)
                cleanup("2 receiving a frame from decoder");
            else
                cleanup("3 receiving a frame from decoder");
            //提取p帧，和i帧
//            if ((frame->pict_type == AV_PICTURE_TYPE_I) || (frame->pict_type == AV_PICTURE_TYPE_P)) {
//                printf("this is i PICTURE\n");
//                //正在解码
//                if (got_picture_ptr) {
//                    //frame->yuvFrame，转为指定的YUV420P像素帧
//                    sws_scale(sws_ctx, (const uint8_t *const *) frame->data, frame->linesize, 0,
//                              frame->height, yuvFrame->data, yuvFrame->linesize);
//                    //计算视频数据总大小
//                    int y_size = pCodeCtx->width * pCodeCtx->height;
//                    //AVFrame->YUV，由于YUV的比例是4:1:1
//                    fwrite(yuvFrame->data[0], 1, y_size, fp_yuv);
//                    fwrite(yuvFrame->data[1], 1, y_size / 4, fp_yuv);
//                    fwrite(yuvFrame->data[2], 1, y_size / 4, fp_yuv);
//                }
//            }

            //提取b帧
            /*
            if(frame->pict_type == AV_PICTURE_TYPE_B)
            {
                //正在解码
                if (got_picture_ptr)
                {
                    //frame->yuvFrame，转为指定的YUV420P像素帧
                    sws_scale(sws_ctx, (const uint8_t *const *) frame->data, frame->linesize, 0,
                              frame->height, yuvFrame->data, yuvFrame->linesize);
                    //计算视频数据总大小
                    int y_size = pCodeCtx->width * pCodeCtx->height;
                    //AVFrame->YUV，由于YUV的比例是4:1:1
                    fwrite(yuvFrame->data[0], 1, y_size, fp_yuv);
                    fwrite(yuvFrame->data[1], 1, y_size / 4, fp_yuv);
                    fwrite(yuvFrame->data[2], 1, y_size / 4, fp_yuv);
                }
            }
            */

        }
    }


    cleanup(NULL);
    return 0;
}