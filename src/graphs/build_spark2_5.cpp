#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph* llm_build_context::build_spark2_5() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : nullptr;

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Spark-X2.5 alternates full-attention and sliding-window-attention layers with
        // layer-dependent RoPE (different rotated dimension and RoPE base per layer type).
        const bool is_swa = KQ_mask_swa != nullptr && hparams.swa_layers[il];
        ggml_tensor * this_KQ_mask = is_swa ? KQ_mask_swa : KQ_mask;
        const int this_n_swa = is_swa ? (int) hparams.n_swa : 0;

        const int64_t n_rot_l      = hparams.rope_n_rot(il);
        const float   freq_base_l  = is_swa ? hparams.rope_freq_base_train_swa  : freq_base;
        const float   freq_scale_l = is_swa ? hparams.rope_freq_scale_train_swa : freq_scale;

        const int64_t n_head_i = hparams.n_head(il);

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // input to the head-wise attention output gate
        ggml_tensor * attn_inp = cur;

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = llm_build_mul_mat_qkv(gf, cur,
                    model.layers[il].wqkv, model.layers[il].bqkv,
                    model.layers[il].wqk,  model.layers[il].bqk,
                    model.layers[il].wq,   model.layers[il].bq,
                    model.layers[il].wk,   model.layers[il].bk,
                    model.layers[il].wv,   model.layers[il].bv,
                    model.layers[il].attn_q_norm, model.layers[il].attn_k_norm, 0, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur", il);

            // attention output without the output projection, which is applied after the gate
            cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    nullptr, nullptr,
                    Kcur, Vcur, Qcur, this_KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il,
                    nullptr, this_n_swa);

            // head-wise sigmoid attention output gate
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv_gate, attn_inp);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate", il);

            cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head, n_head_i, n_tokens);
            gate = ggml_reshape_3d(ctx0, gate, 1,            n_head_i, n_tokens);
            cur  = ggml_mul(ctx0, cur, gate);
            cur  = ggml_reshape_2d(ctx0, cur, n_embd_head*n_head_i, n_tokens);
            cb(cur, "attn_gated", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network (GELU)
        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
