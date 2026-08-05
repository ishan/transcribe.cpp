// sortformer_stream_parity.c — the acceptance test for the push-audio path.
//
// Claim under test: feeding a recording through transcribe_stream_feed in
// ARBITRARY increments must produce exactly the speaker segments that
// transcribe_run produces over the same recording in one call.
//
// That is the whole contract. The streaming core (AOSC speaker cache + FIFO)
// was always incremental; what the STREAM slot adds is the feature-side
// bookkeeping — holding enough PCM to compute each mel frame exactly once its
// analysis window has closed, and dispatching a chunk only when its full right
// context is final. Both are easy to get subtly wrong and neither fails
// loudly, so the test compares every segment row rather than a summary.
//
// Run:
//     transcribe-sortformer-stream-parity model.gguf audio.wav [chunk_ms ...]
//
// Defaults to a spread of feed sizes, including ones far smaller and far
// larger than the model's internal chunk, since a correct implementation must
// be insensitive to how the caller happens to packetize its audio.

#include "transcribe.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEGS 65536

struct seg_list {
    int     n;
    int64_t t0[MAX_SEGS];
    int64_t t1[MAX_SEGS];
    int32_t spk[MAX_SEGS];
};

static float * load_wav_mono(const char * path, int * out_n_samples, unsigned int * out_rate) {
    unsigned int channels = 0, rate = 0;
    drwav_uint64 frames = 0;
    float * interleaved = drwav_open_file_and_read_pcm_frames_f32(path, &channels, &rate, &frames, NULL);
    if (interleaved == NULL) {
        return NULL;
    }
    *out_rate = rate;
    if (channels <= 1) {
        *out_n_samples = (int) frames;
        return interleaved;
    }
    float * mono = (float *) malloc((size_t) frames * sizeof(float));
    for (drwav_uint64 i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            acc += interleaved[i * channels + c];
        }
        mono[i] = acc / (float) channels;
    }
    drwav_free(interleaved, NULL);
    *out_n_samples = (int) frames;
    return mono;
}

static int collect(struct transcribe_session * s, struct seg_list * out) {
    out->n = transcribe_n_speaker_segments(s);
    if (out->n > MAX_SEGS) {
        fprintf(stderr, "error: %d segments exceeds MAX_SEGS\n", out->n);
        return 0;
    }
    for (int i = 0; i < out->n; ++i) {
        struct transcribe_speaker_segment seg;
        transcribe_speaker_segment_init(&seg);
        if (transcribe_get_speaker_segment(s, i, &seg) != TRANSCRIBE_OK) {
            return 0;
        }
        out->t0[i]  = seg.t0_ms;
        out->t1[i]  = seg.t1_ms;
        out->spk[i] = seg.speaker_id;
    }
    return 1;
}

// Report the FIRST divergence rather than a count: with a stateful speaker
// cache, one bad chunk shifts everything after it, so the first mismatch is
// the only informative one.
static int compare(const struct seg_list * a, const struct seg_list * b, const char * label) {
    if (a->n != b->n) {
        printf("  %-10s FAIL  segment count %d (batch) vs %d (stream)\n", label, a->n, b->n);
        return 0;
    }
    for (int i = 0; i < a->n; ++i) {
        if (a->t0[i] != b->t0[i] || a->t1[i] != b->t1[i] || a->spk[i] != b->spk[i]) {
            printf("  %-10s FAIL  row %d: batch spk%d [%lld,%lld] vs stream spk%d [%lld,%lld]\n", label, i,
                   a->spk[i], (long long) a->t0[i], (long long) a->t1[i], b->spk[i], (long long) b->t0[i],
                   (long long) b->t1[i]);
            return 0;
        }
    }
    printf("  %-10s PASS  %d segments identical\n", label, a->n);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf audio.wav [chunk_ms ...]\n", argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const char * wav_path   = argv[2];

    int default_chunks[] = { 100, 500, 1000, 5000, 20000 };
    int n_chunks         = (argc > 3) ? argc - 3 : (int) (sizeof(default_chunks) / sizeof(int));
    int chunks[64];
    for (int i = 0; i < n_chunks && i < 64; ++i) {
        chunks[i] = (argc > 3) ? atoi(argv[3 + i]) : default_chunks[i];
    }

    int          n_samples = 0;
    unsigned int rate      = 0;
    float *      pcm       = load_wav_mono(wav_path, &n_samples, &rate);
    if (pcm == NULL || rate != 16000) {
        fprintf(stderr, "error: need a readable 16 kHz WAV (%s)\n", wav_path);
        free(pcm);
        return 1;
    }
    printf("audio: %.2f s\n", (double) n_samples / 16000.0);

    struct transcribe_session * session = NULL;
    if (transcribe_open(model_path, NULL, NULL, &session) != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_open failed\n");
        free(pcm);
        return 1;
    }

    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    transcribe_model_get_capabilities(transcribe_get_model(session), &caps);
    printf("supports_streaming: %s\n", caps.supports_streaming ? "yes" : "no");
    if (!caps.supports_streaming) {
        fprintf(stderr, "error: model does not advertise streaming\n");
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }

    // --- reference: one whole-file transcribe_run ---
    static struct seg_list ref;
    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    int64_t t_batch0 = 0, t_batch1 = 0;
    t_batch0 = (int64_t) clock();
    if (transcribe_run(session, pcm, n_samples, &rp) != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_run failed\n");
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }
    t_batch1 = (int64_t) clock();
    if (!collect(session, &ref)) {
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }
    printf("batch: %d segments in %.2f s\n\n", ref.n,
           (double) (t_batch1 - t_batch0) / (double) CLOCKS_PER_SEC);

    // --- streaming, at several feed sizes ---
    int all_pass = 1;
    for (int c = 0; c < n_chunks; ++c) {
        const int chunk_samples = chunks[c] * 16000 / 1000;
        if (chunk_samples <= 0) {
            continue;
        }
        if (transcribe_stream_begin(session, NULL, NULL) != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_begin failed\n");
            all_pass = 0;
            break;
        }
        const int64_t t0   = (int64_t) clock();
        int           pos  = 0;
        int           fail = 0;
        while (pos < n_samples) {
            int take = chunk_samples;
            if (pos + take > n_samples) {
                take = n_samples - pos;
            }
            struct transcribe_stream_update upd;
            transcribe_stream_update_init(&upd);
            if (transcribe_stream_feed(session, pcm + pos, take, &upd) != TRANSCRIBE_OK) {
                fprintf(stderr, "error: stream_feed failed at %d\n", pos);
                fail = 1;
                break;
            }
            pos += take;
        }
        if (fail) {
            all_pass = 0;
            continue;
        }
        struct transcribe_stream_update fin;
        transcribe_stream_update_init(&fin);
        if (transcribe_stream_finalize(session, &fin) != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_finalize failed\n");
            all_pass = 0;
            continue;
        }
        const int64_t t1 = (int64_t) clock();

        static struct seg_list got;
        if (!collect(session, &got)) {
            all_pass = 0;
            continue;
        }
        char label[32];
        snprintf(label, sizeof(label), "%d ms", chunks[c]);
        if (!compare(&ref, &got, label)) {
            all_pass = 0;
        }
        printf("             %.2f s wall\n", (double) (t1 - t0) / (double) CLOCKS_PER_SEC);
    }

    transcribe_session_free(session);
    free(pcm);
    printf("\n%s\n", all_pass ? "PARITY OK" : "PARITY FAILED");
    return all_pass ? 0 : 1;
}
