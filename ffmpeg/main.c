#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

char* input_path = "../video/01.mp4";

void cleanup(){

}

int main(int argc, char **argv) {
    //2、打开视频文件
    AVFormatContext *pFormatCtx = avformat_alloc_context();
    if ((avformat_open_input(&pFormatCtx, input_path, NULL, NULL)) < 0)
    {
        perror("Cannot open input file");
        exit(1);
    }

    //3、获取视频信息
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
    {
        perror("Cannot find stream");
        if(pFormatCtx)
            avformat_free_context(pFormatCtx);
        avformat_close_input(&pFormatCtx);
        exit(2);
    }

    //4、找到视频流的位置
    int video_stream_index = -1;
    int i = 0;
    for (; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1){
        perror("Cannot find stream index");
        exit(1);
    }


    return 0;
}