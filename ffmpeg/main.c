#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>

#include <dirent.h>
#include <sys/stat.h>
#include <zconf.h>

#define BUFF_SIZE 1024

char *input_dir = "../video/%s";
char *filename = "01.mp4";
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
struct SwsContext *sws_ctx = NULL;
AVFrame *yuvFrame = NULL;
uint8_t *out_buffer = NULL;

int remove_directory(const char *path);
void pgm_save(unsigned char *buf, int wrap, char *filename);


void cleanup(char *msg) {
    if (pFormatCtx)
        avformat_close_input(&pFormatCtx);
    if (codec_ctx)
        avcodec_free_context(&codec_ctx);
    if (packet)
        av_packet_unref(packet);
    if (frame)
        av_frame_free(&frame);
    if (yuvFrame)
        av_frame_free(&yuvFrame);
    if (out_buffer)
        av_free(out_buffer);
    if (msg) {
        perror(msg);
        exit(1);
    }
    remove_directory(save_path);
}

int extract_frame(AVFrame *frame) {
    if (frame->pict_type == AV_PICTURE_TYPE_I) {
        snprintf(buf, BUFF_SIZE, "%s/%s-%d.yuv", save_path, filename, codec_ctx->frame_number);
//        pgm_save(frame->data[0], frame->linesize[0], buf);
        FILE* fp_yuv = fopen(buf, "w");
        //frame->yuvFrame，转为指定的YUV420P像素帧
        sws_scale(sws_ctx, (const uint8_t *const *) frame->data, frame->linesize, 0,
                  frame->height, yuvFrame->data, yuvFrame->linesize);
        //计算视频数据总大小
        int y_size = width * height;
        //AVFrame->YUV，由于YUV的比例是4:1:1
        fwrite(yuvFrame->data[0], 1, y_size, fp_yuv);
        fwrite(yuvFrame->data[1], 1, y_size / 4, fp_yuv);
        fwrite(yuvFrame->data[2], 1, y_size / 4, fp_yuv);

    }
    return 0;
}

void init() {
    if (snprintf(input_path, BUFF_SIZE, input_dir, filename) < 0)
        cleanup("Error in snprintf");
    if (snprintf(save_path, BUFF_SIZE, "%s-IFrame", input_path) < 0)
        cleanup("Error in snprintf");
    remove_directory(save_path);
    if (mkdir(save_path, 0777) < 0)
        cleanup("Error in mkdir");
}

int main(int argc, char **argv) {

    init();

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

    // change format
    yuvFrame = av_frame_alloc();
    sws_ctx = sws_getContext(
            width,
            height,
            codec_ctx->pix_fmt,
            width,
            height,
            AV_PIX_FMT_YUV420P, SWS_BILINEAR, NULL, NULL, NULL);
    out_buffer = (uint8_t *) av_malloc(av_image_get_buffer_size(AV_PIX_FMT_YUV420P, width, height, 1));
    av_image_fill_arrays(yuvFrame->data, yuvFrame->linesize, out_buffer, AV_PIX_FMT_YUV420P, width, height, 1);


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
    return 0;
}


int remove_directory(const char *path) {
    DIR *d = opendir(path);
    size_t path_len = strlen(path);
    int r = -1;

    if (d) {
        struct dirent *p;
        r = 0;
        while (!r && (p = readdir(d))) {
            int r2 = -1;
            char *buf;
            size_t len;

            /* Skip the names "." and ".." as we don't want to recurse on them. */
            if (!strcmp(p->d_name, ".") || !strcmp(p->d_name, "..")) {
                continue;
            }

            len = path_len + strlen(p->d_name) + 2;
            buf = malloc(len);

            if (buf) {
                struct stat statbuf;
                snprintf(buf, len, "%s/%s", path, p->d_name);
                if (!stat(buf, &statbuf)) {
                    if (S_ISDIR(statbuf.st_mode)) {
                        r2 = remove_directory(buf);
                    } else {
                        r2 = unlink(buf);
                    }
                }
                free(buf);
            }
            r = r2;
        }
        closedir(d);
    }

    if (!r)
        r = rmdir(path);

    return r;
}


void pgm_save(unsigned char *buf, int wrap, char *filename) {
    FILE *f;
    int i;
    f = fopen(filename, "w");
    fprintf(f, "P5\n%d %d\n%d\n", width, height, 255);
    for (i = 0; i < height; i++)
        fwrite(buf + i * wrap, 1, width, f);
    fclose(f);
}
