#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#include "io.h"
#include "packet_queue.h"

#define BUFF_SIZE 1024

char *cmd = "split";
char *input_dir = "../video";
char *filename = "02.mp4";
char save_path[BUFF_SIZE];
char input_path[BUFF_SIZE];
char output_path[BUFF_SIZE];
char buf[BUFF_SIZE];
int width = 0;
int height = 0;
int ret = 0;
int i_video_stream_index = -1;
int o_video_stream_index = -1;
int frame_cnt = 1;
AVFormatContext *ifmt_ctx = NULL;
AVFormatContext *ofmt_ctx = NULL;
AVCodec *i_pCodec = NULL;
AVCodec *o_pCodec = NULL;
AVCodecContext *i_codec_ctx = NULL;
AVCodecContext *o_codec_ctx = NULL;
AVCodecParameters *i_pCodePara = NULL;
AVCodecParameters *o_pCodePara = NULL;
AVPacket *packet = NULL;
AVFrame *frame = NULL;
AVOutputFormat *ofmt = NULL;
AVCodec *video_codec = NULL;
AVCodecContext *video_codec_ctx = NULL;
Packet_Queue pk_queue1;
Packet_Queue pk_queue2;

void cleanup(char *msg) {
    if (ifmt_ctx)
        avformat_close_input(&ifmt_ctx);
    if (ofmt_ctx)
        avformat_close_input(&ofmt_ctx);
    if (i_codec_ctx)
        avcodec_free_context(&i_codec_ctx);
    if (o_codec_ctx)
        avcodec_free_context(&o_codec_ctx);
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
    if (snprintf(output_path, BUFF_SIZE, "%s/out-%s", input_dir, filename) < 0)
        cleanup("Error in snprintf");

    printf("%s\n", input_path);
    printf("%s\n", output_path);
}

void save_frame_as_jpeg(AVFrame *pFrame, char *filename) {
    AVCodec *jpegCodec = avcodec_find_encoder(AV_CODEC_ID_JPEG2000);
    if (!jpegCodec)
        cleanup("Error in finding jpeg decoder");
    AVCodecContext *jpegContext = avcodec_alloc_context3(jpegCodec);
    if (!jpegContext)
        cleanup("Error in create jpeg context");
    jpegContext->pix_fmt = i_codec_ctx->pix_fmt;
    jpegContext->height = height;
    jpegContext->width = width;
    jpegContext->time_base = i_codec_ctx->time_base;
    if (avcodec_open2(jpegContext, jpegCodec, NULL) < 0)
        cleanup("Error in opening jpeg decoder");
    AVPacket *jpeg_packet = av_packet_alloc();

    if (avcodec_send_frame(jpegContext, pFrame) < 0)
        cleanup("Error in sending frame");
    if (avcodec_receive_packet(jpegContext, jpeg_packet) < 0)
        cleanup("Error in receiving packet");

    FILE *JPEGFile = fopen(filename, "wb");
    fwrite(jpeg_packet->data, 1, jpeg_packet->size, JPEGFile);
    fclose(JPEGFile);

    avcodec_close(jpegContext);
    av_packet_unref(jpeg_packet);
}

void video_codec_init() {
    video_codec = avcodec_find_encoder(ofmt_ctx->streams[o_video_stream_index]->codecpar->codec_id);
    video_codec_ctx = avcodec_alloc_context3(video_codec);
    video_codec_ctx->time_base = i_codec_ctx->time_base;
    video_codec_ctx->pix_fmt = i_codec_ctx->pix_fmt;
    video_codec_ctx->width = i_codec_ctx->width;
    video_codec_ctx->height = i_codec_ctx->height;

    if (avcodec_open2(video_codec_ctx, video_codec, NULL) < 0)
        cleanup("Error in openingsss codec");
}

void write_output_video_packet(AVPacket* output_packet, AVPacket* original_packet){
    output_packet->dts = original_packet->dts;
    output_packet->pts = original_packet->pts;
    output_packet->duration = original_packet->duration;
    output_packet->size = original_packet->size;
    output_packet->pos = original_packet->pos;
    output_packet->stream_index = original_packet->stream_index;
    if (av_interleaved_write_frame(ofmt_ctx, output_packet) < 0)
        cleanup("error in write packet");
}

AVPacket *load_jpeg_as_packet(char *filename, AVPacket *original_packet) {


    /*
     *
     *
     * 这他妈有bug，我找了半天也没找到
     * 这里需要做的就是把磁盘上的图片重新加载进内存，还原成一个frame，把原来pFrame里面的内容替换掉
     */

    /*
     *
     *
     * 改来改去还他妈的是这里有bug
     */



    AVCodec *jpegCodec = avcodec_find_decoder(AV_CODEC_ID_JPEG2000);
    if (!jpegCodec)
        cleanup("Error in finding jpeg decoder");
    AVCodecContext *jpegContext = avcodec_alloc_context3(jpegCodec);
    if (!jpegContext)
        cleanup("Error in create jpeg context");
    jpegContext->pix_fmt = i_codec_ctx->pix_fmt;
    jpegContext->height = height;
    jpegContext->width = width;
    if (avcodec_open2(jpegContext, jpegCodec, NULL) < 0)
        cleanup("Error in opening jpeg decoder");

    int size = get_file_size(filename);
    AVPacket *jpeg_packet = av_packet_alloc();
    if (av_new_packet(jpeg_packet, size) < 0)
        cleanup("Error in allocating packet");

    FILE *JPEGFile = fopen(filename, "rb");
    int n = fread(jpeg_packet->data, 1, jpeg_packet->size, JPEGFile);
    if (n != size)
        cleanup("Error in reading jpeg file");
    fclose(JPEGFile);
    avcodec_close(video_codec_ctx);

    // 4. read jpeg file from disk into a jpeg_packet
    if (avcodec_send_packet(jpegContext, jpeg_packet) < 0)
        cleanup("Error in sending packet");
    AVFrame *jpeg_frame = av_frame_alloc();
    // 5. decode jpeg_packet into new_input_frame
    if (avcodec_receive_frame(jpegContext, jpeg_frame) < 0)
        cleanup("cannot receive jpeg packet");
    avcodec_close(jpegContext);

    // create a video encoder
    video_codec_init();
    // 6. encode new_input_frame into output_video_packet
    if (avcodec_send_frame(video_codec_ctx, jpeg_frame) < 0)
        cleanup("Error in sending jpeg frame");
    // enter drain mode
    if (avcodec_send_frame(video_codec_ctx, NULL) < 0)
        cleanup("Error in entering video encoding drain mode");
    AVPacket *video_packet = av_packet_alloc();
    if (avcodec_receive_packet(video_codec_ctx, video_packet) < 0)
        cleanup("cannot receive video packet");


    // 7. throw output_video_packet into output_stream
    write_output_video_packet(video_packet, original_packet);




    av_frame_free(&jpeg_frame);
    return video_packet;


}

void split_process_frame(AVFrame *frame) {
    printf("%d\n", frame_cnt);
    if (frame->pict_type == AV_PICTURE_TYPE_I) {
        snprintf(buf, BUFF_SIZE, "%s/%s-%d.jpg", save_path, filename, frame_cnt);
        save_frame_as_jpeg(frame, buf);

    }
    frame_cnt++;

}


int merge_process_frame(AVFrame *frame) {
    AVPacket *original_packet = pop_packet(&pk_queue1);
    if (!original_packet)
        cleanup("empty packet queue");

    // 2. is input_frame.type == I_FRAME goto 4
    printf("%d\n", frame_cnt);
    if (frame->pict_type == AV_PICTURE_TYPE_I) {
        snprintf(buf, BUFF_SIZE, "%s/%s-%d.jpg", save_path, filename, frame_cnt);
        load_jpeg_as_packet(buf, original_packet);
    } else {
        // 3. throw input_video_packet to av_interleaved_write_frame
        if (av_interleaved_write_frame(ofmt_ctx, original_packet) < 0)
            cleanup("error in write frame");
    }


    av_packet_unref(original_packet);
    frame_cnt++;
}

int split() {
    // remove old data
    remove_directory(save_path);
    if (mkdir(save_path, 0777) < 0)
        cleanup("Error in mkdir");


    //2、打开视频文件
    ifmt_ctx = avformat_alloc_context();
    if ((avformat_open_input(&ifmt_ctx, input_path, NULL, NULL)) < 0)
        cleanup("Error in opening input file");

    //3、获取视频信息
    if (avformat_find_stream_info(ifmt_ctx, NULL) < 0)
        cleanup("Error in finding stream");

    //4、找到视频流的位置
    i_video_stream_index = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &i_pCodec, 0);
    if (i_video_stream_index == AVERROR_STREAM_NOT_FOUND)
        cleanup("Error in finding stream index");
    else if (i_video_stream_index == AVERROR_DECODER_NOT_FOUND)
        cleanup("Error in finding decoder");

    //5、获取AVCodecParameters
    i_pCodePara = ifmt_ctx->streams[i_video_stream_index]->codecpar;
    width = i_pCodePara->width;
    height = i_pCodePara->height;

    //6、打开解码器
    i_codec_ctx = avcodec_alloc_context3(i_pCodec);
    avcodec_parameters_to_context(i_codec_ctx, i_pCodePara);
    if (avcodec_open2(i_codec_ctx, i_pCodec, NULL) < 0)
        cleanup("Error in opening codec");

    //7、解析每一帧数据
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    // read all frames and send them into decoder

    while (av_read_frame(ifmt_ctx, packet) >= 0) {
        if (packet->stream_index == i_video_stream_index) {
            if (avcodec_send_packet(i_codec_ctx, packet) < 0)
                cleanup("Error in sending a packet for decoding");
            while ((ret = avcodec_receive_frame(i_codec_ctx, frame)) >= 0)
                split_process_frame(frame);
            if (ret == AVERROR(EAGAIN))
                continue;
            else if (ret < 0)
                cleanup("Error in receiving a packet from decoder");
        }
    }
    // send flush packet, enter draining mode.
    avcodec_send_packet(i_codec_ctx, NULL);
    while ((ret = avcodec_receive_frame(i_codec_ctx, frame)) >= 0)
        split_process_frame(frame);
    if (ret != AVERROR_EOF)
        cleanup("Error in draining decoder stream");

    cleanup(NULL);
    return 1;
}

int merge() {
    if (avformat_open_input(&ifmt_ctx, input_path, 0, 0) < 0)
        cleanup("Could not open input file");

    // Extract streams description
    if ((ret = avformat_find_stream_info(ifmt_ctx, 0)) < 0)
        cleanup("Failed to retrieve input stream information");

    avformat_alloc_output_context2(&ofmt_ctx, NULL, NULL, output_path);
    if (!ofmt_ctx) {
        cleanup("Could not create output context\n");
    }
    ofmt = ofmt_ctx->oformat;

    // Allocating output streams
    for (int i = 0; i < ifmt_ctx->nb_streams; i++) {
        AVStream *in_stream = ifmt_ctx->streams[i];
        AVStream *out_stream = avformat_new_stream(ofmt_ctx, in_stream->codec->codec);
        if (!out_stream) {
            cleanup("Failed allocating output stream\n");
        }
        ret = avcodec_copy_context(out_stream->codec, in_stream->codec);
        if (ret < 0) {
            cleanup("Failed to copy context from input to output stream codec context\n");
        }
        out_stream->codec->codec_tag = 0;
        if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            out_stream->codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
    }

    // Open output file
    if (!(ofmt->flags & AVFMT_NOFILE)) {
        ret = avio_open(&ofmt_ctx->pb, output_path, AVIO_FLAG_WRITE);
        if (ret < 0) {
            cleanup("Could not open output file");
        }
    }

    //header
    ret = avformat_write_header(ofmt_ctx, NULL);
    if (ret < 0) {
        cleanup("Error in writing header\n");
    }

    // input codec
    i_video_stream_index = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &i_pCodec, 0);
    if (i_video_stream_index == AVERROR_STREAM_NOT_FOUND)
        cleanup("Error in finding stream index");
    else if (i_video_stream_index == AVERROR_DECODER_NOT_FOUND)
        cleanup("Error in finding decoder");

    i_pCodePara = ifmt_ctx->streams[i_video_stream_index]->codecpar;
    width = i_pCodePara->width;
    height = i_pCodePara->height;

    i_codec_ctx = avcodec_alloc_context3(i_pCodec);
    avcodec_parameters_to_context(i_codec_ctx, i_pCodePara);
    if (avcodec_open2(i_codec_ctx, i_pCodec, NULL) < 0)
        cleanup("Error in opening codec");
    // output codec
    o_video_stream_index = av_find_best_stream(ofmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &o_pCodec, 0);
    if (o_video_stream_index == AVERROR_STREAM_NOT_FOUND)
        cleanup("Error in finding stream index");
    else if (o_video_stream_index == AVERROR_DECODER_NOT_FOUND)
        cleanup("Error in finding decoder");
    o_pCodePara = ofmt_ctx->streams[o_video_stream_index]->codecpar;

    o_codec_ctx = avcodec_alloc_context3(o_pCodec);
    avcodec_parameters_to_context(o_codec_ctx, o_pCodePara);
    if (avcodec_open2(o_codec_ctx, o_pCodec, NULL) < 0)
        cleanup("Error in opening codec");

    frame = av_frame_alloc();
    packet_queue_alloc(&pk_queue1, 1000);
    while (1) {
        packet = av_packet_alloc();
        ret = av_read_frame(ifmt_ctx, packet);
        if (ret < 0)
            break;

        AVStream *in_stream, *out_stream;
        in_stream = ifmt_ctx->streams[packet->stream_index];
        out_stream = ofmt_ctx->streams[packet->stream_index];

        // copy packet
        packet->pts = av_rescale_q_rnd(packet->pts, in_stream->time_base, out_stream->time_base,
                                       AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
        packet->dts = av_rescale_q_rnd(packet->dts, in_stream->time_base, out_stream->time_base,
                                       AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
        packet->duration = av_rescale_q(packet->duration, in_stream->time_base, out_stream->time_base);

        if (packet->stream_index == i_video_stream_index) {
            // video stream
            // 1. input_video_packet -> input_frame
            if (avcodec_send_packet(i_codec_ctx, packet) < 0)
                cleanup("error in phase 1");
            if (push_packet(&pk_queue1, packet) < 0)
                cleanup("cannot push into packet queue");
            while ((ret = avcodec_receive_frame(i_codec_ctx, frame)) >= 0)
                merge_process_frame(frame);
            if (ret == AVERROR(EAGAIN))
                continue;
            else
                cleanup("Error in receiving a packet from decoder");
        } else {
            // other stream
            if (av_interleaved_write_frame(ofmt_ctx, packet) < 0)
                cleanup("error in writing other stream");
        }

    }

    if (avcodec_send_packet(i_codec_ctx, NULL) < 0)
        cleanup("error enter drain mode");
    while (avcodec_receive_frame(i_codec_ctx, frame) >= 0)
        merge_process_frame(frame);
    if (ret != AVERROR_EOF)
        cleanup("Error in draining decoder stream");

    packet_queue_free(&pk_queue1);

    if (av_interleaved_write_frame(ofmt_ctx, NULL) < 0)
        cleanup("error in flush");

    if ((av_write_trailer(ofmt_ctx) < 0))
        cleanup("error in write trailer");


    cleanup(NULL);


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
        split();

    } else {

        merge();
    }
}





