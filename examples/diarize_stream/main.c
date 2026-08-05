// diarize_stream.c — streaming Sortformer diarization with dynamic windowing.
//
// Sortformer has a hard architectural cap of four concurrent speakers, and it
// gives no signal when a fifth appears: the fifth is silently absorbed into an
// existing slot. Because that failure is invisible, slot 4 is treated as a
// TRIPWIRE rather than a usable slot — the moment it activates, the window is
// cut immediately BEFORE that speaker's first turn and a fresh window is begun
// with them as its first speaker. Every window therefore holds at most three
// speakers and absorption is structurally impossible.
//
// This is why the push-audio entry point matters. On the batch API a cut costs
// a re-diarization of the whole remaining recording, because finding the next
// cut means running the model over everything after this one. Here a cut is a
// state reset plus a re-feed of the audio between the cut point and where the
// stream had already read to — bounded by one chunk plus lookahead, not by the
// length of the meeting.
//
// Speaker ids are per-window and arrival-ordered, so they carry no meaning
// across a cut. Linking them into real identities is the caller's job, done
// with voiceprints downstream; this tool reports the window index alongside
// every turn so that linking has something to key on.
//
// Output: one JSON object per line.
//     {"window":0,"speaker":1,"t0_ms":1200,"t1_ms":4300}
//     {"event":"cut","window":0,"at_ms":193840,"reason":"slot-4 activation"}
//
// Run:
//     transcribe-diarize-stream model.gguf audio.wav [feed_ms=1000]

#include "transcribe.h"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEGS 65536
#define SPEAKER_CAP 4

static float * load_wav_mono(const char * path, int * out_n, unsigned int * out_rate) {
    unsigned int channels = 0, rate = 0;
    drwav_uint64 frames = 0;
    float * inter = drwav_open_file_and_read_pcm_frames_f32(path, &channels, &rate, &frames, NULL);
    if (inter == NULL) {
        return NULL;
    }
    *out_rate = rate;
    if (channels <= 1) {
        *out_n = (int) frames;
        return inter;
    }
    float * mono = (float *) malloc((size_t) frames * sizeof(float));
    for (drwav_uint64 i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            acc += inter[i * channels + c];
        }
        mono[i] = acc / (float) channels;
    }
    drwav_free(inter, NULL);
    *out_n = (int) frames;
    return mono;
}

// Earliest start time of the tripwire slot in the current window, or -1.
// Arrival-ordered ids make "speaker_id == SPEAKER_CAP" exactly "the fourth
// distinct voice this window has heard".
static int64_t tripwire_at(struct transcribe_session * s) {
    const int n = transcribe_n_speaker_segments(s);
    int64_t   best = -1;
    for (int i = 0; i < n && i < MAX_SEGS; ++i) {
        struct transcribe_speaker_segment seg;
        transcribe_speaker_segment_init(&seg);
        if (transcribe_get_speaker_segment(s, i, &seg) != TRANSCRIBE_OK) {
            continue;
        }
        if (seg.speaker_id == SPEAKER_CAP && (best < 0 || seg.t0_ms < best)) {
            best = seg.t0_ms;
        }
    }
    return best;
}

// Emit every turn that starts before `limit_ms` (or all of them when limit_ms
// is negative), offsetting window-relative times back onto the recording.
static void emit(struct transcribe_session * s, int window, int64_t offset_ms, int64_t limit_ms) {
    const int n = transcribe_n_speaker_segments(s);
    for (int i = 0; i < n && i < MAX_SEGS; ++i) {
        struct transcribe_speaker_segment seg;
        transcribe_speaker_segment_init(&seg);
        if (transcribe_get_speaker_segment(s, i, &seg) != TRANSCRIBE_OK || seg.speaker_id == 0) {
            continue;
        }
        if (limit_ms >= 0 && seg.t0_ms >= limit_ms) {
            continue;
        }
        int64_t t1 = seg.t1_ms;
        if (limit_ms >= 0 && t1 > limit_ms) {
            t1 = limit_ms;
        }
        printf("{\"window\":%d,\"speaker\":%d,\"t0_ms\":%lld,\"t1_ms\":%lld}\n", window, seg.speaker_id,
               (long long) (offset_ms + seg.t0_ms), (long long) (offset_ms + t1));
    }
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s model.gguf audio.wav [feed_ms=1000]\n", argv[0]);
        return 2;
    }
    const int feed_ms = (argc == 4) ? atoi(argv[3]) : 1000;
    if (feed_ms <= 0) {
        fprintf(stderr, "error: feed_ms must be > 0\n");
        return 2;
    }

    int          n_samples = 0;
    unsigned int rate      = 0;
    float *      pcm       = load_wav_mono(argv[2], &n_samples, &rate);
    if (pcm == NULL || rate != 16000) {
        fprintf(stderr, "error: need a readable 16 kHz mono WAV\n");
        free(pcm);
        return 1;
    }

    struct transcribe_session * session = NULL;
    if (transcribe_open(argv[1], NULL, NULL, &session) != TRANSCRIBE_OK) {
        fprintf(stderr, "error: transcribe_open failed\n");
        free(pcm);
        return 1;
    }
    struct transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    transcribe_model_get_capabilities(transcribe_get_model(session), &caps);
    if (!caps.supports_streaming) {
        fprintf(stderr, "error: model does not advertise streaming\n");
        transcribe_session_free(session);
        free(pcm);
        return 1;
    }

    const int feed_samples = feed_ms * 16000 / 1000;
    int       window       = 0;
    int       window_start = 0;  // sample index where this window begins
    int       pos          = 0;  // next sample to feed
    int       rc           = 0;

    if (transcribe_stream_begin(session, NULL, NULL) != TRANSCRIBE_OK) {
        fprintf(stderr, "error: stream_begin failed\n");
        rc = 1;
        goto done;
    }

    while (pos < n_samples) {
        int take = feed_samples;
        if (pos + take > n_samples) {
            take = n_samples - pos;
        }
        struct transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        if (transcribe_stream_feed(session, pcm + pos, take, &upd) != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_feed failed at sample %d\n", pos);
            rc = 1;
            goto done;
        }
        pos += take;

        if (!upd.result_changed) {
            continue;
        }
        const int64_t cut_rel = tripwire_at(session);
        if (cut_rel < 0) {
            continue;
        }

        // Cut before the fourth speaker's first turn and restart there. Only
        // the audio between the cut and where we had read to is re-fed, which
        // is what makes this affordable to do repeatedly.
        const int64_t offset_ms = (int64_t) window_start * 1000 / 16000;
        emit(session, window, offset_ms, cut_rel);
        printf("{\"event\":\"cut\",\"window\":%d,\"at_ms\":%lld,\"reason\":\"slot-4 activation\"}\n", window,
               (long long) (offset_ms + cut_rel));
        fflush(stdout);

        const int cut_sample = window_start + (int) (cut_rel * 16000 / 1000);
        if (cut_sample <= window_start) {
            fprintf(stderr, "warning: degenerate cut at window start; continuing without reset\n");
            continue;
        }
        ++window;
        window_start = cut_sample;
        // The session is still ACTIVE, and begin refuses to re-enter from
        // there. Reset drops the AOSC cache and FIFO and returns the
        // lifecycle to IDLE — that discard IS the window boundary.
        transcribe_stream_reset(session);
        if (transcribe_stream_begin(session, NULL, NULL) != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_begin (window %d) failed\n", window);
            rc = 1;
            goto done;
        }
        pos = window_start;  // re-feed from the cut
    }

    {
        struct transcribe_stream_update fin;
        transcribe_stream_update_init(&fin);
        if (transcribe_stream_finalize(session, &fin) != TRANSCRIBE_OK) {
            fprintf(stderr, "error: stream_finalize failed\n");
            rc = 1;
            goto done;
        }
        emit(session, window, (int64_t) window_start * 1000 / 16000, -1);
    }

done:
    transcribe_session_free(session);
    free(pcm);
    return rc;
}
