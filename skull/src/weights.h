#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "tensor.h"
#include "tokenizer.h"
#include "transformer.h"

// ============================================================
//  SKULL WEIGHTS  —  Gewichte speichern und laden
//
//  Dateiformat (little-endian, IEEE-754 double). Zwei Modellarten:
//
//  A) Bigram-Modell ("SKULL"):
//    "SKULL"                       5 Byte Magic
//    uint64  dim
//    uint64  vocab
//    double  W_embed [vocab * dim]
//    double  W_hidden[dim   * dim]
//    double  W_out   [dim   * vocab]
//
//  B) Transformer ("SKULT"):
//    "SKULT"                       5 Byte Magic
//    uint32  version (= 1)
//    uint64  vocab, dim, context, heads, layers
//    double  params[param_count]   (Reihenfolge siehe transformer.h)
//
//  Beide koennen am Ende (nur bei BPE-Training) tragen:
//    "TOKB"                        4 Byte Magic
//    uint32  n_merges
//    n_merges x ( uint32 len_a, a[len_a], uint32 len_b, b[len_b] )
//
//  Aeltere Dateien ohne TOKB-Abschnitt (Zeichen-Tokenizer) bleiben
//  lesbar. Beim Laden wird die Dateigroesse gegen die Header-Werte
//  geprueft, bevor Speicher angelegt wird.
// ============================================================

struct SkullWeights {
    size_t       dim   = 0;
    size_t       vocab = 0;
    // Bigram-Modell
    FlatVec      W_embed;    // vocab x dim
    FlatVec      W_hidden;   // dim   x dim
    FlatVec      W_out;      // dim   x vocab
    // Transformer
    bool              is_transformer = false;
    TransformerConfig tcfg;
    FlatVec           tparams;
    // Tokenizer (optional)
    bool         has_bpe = false;
    BPETokenizer bpe;
};

namespace weights_detail {

static constexpr uint64_t MAX_DIM    = 1ull << 16;   // 65536
static constexpr uint64_t MAX_VOCAB  = 1ull << 24;   // 16 Mio
static constexpr uint64_t MAX_CONTEXT = 1ull << 13;  // 8192
static constexpr uint64_t MAX_LAYERS  = 1ull << 10;  // 1024
static constexpr uint64_t MAX_MERGES = 1ull << 20;
static constexpr uint64_t MAX_TOKLEN = 1ull << 12;

inline void put_u64(std::ostream& o, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    o.write(reinterpret_cast<const char*>(b), 8);
}
inline void put_u32(std::ostream& o, uint32_t v) {
    unsigned char b[4];
    for (int i = 0; i < 4; ++i) b[i] = (unsigned char)((v >> (8 * i)) & 0xFF);
    o.write(reinterpret_cast<const char*>(b), 4);
}
inline bool get_u64(std::istream& in, uint64_t& v) {
    unsigned char b[8];
    if (!in.read(reinterpret_cast<char*>(b), 8)) return false;
    v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t)b[i] << (8 * i);
    return true;
}
inline bool get_u32(std::istream& in, uint32_t& v) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4)) return false;
    v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)b[i] << (8 * i);
    return true;
}
inline void put_str(std::ostream& o, const std::string& s) {
    put_u32(o, (uint32_t)s.size());
    o.write(s.data(), (std::streamsize)s.size());
}
inline bool get_str(std::istream& in, std::string& s) {
    uint32_t n = 0;
    if (!get_u32(in, n) || n > MAX_TOKLEN) return false;
    s.resize(n);
    return n == 0 || (bool)in.read(&s[0], (std::streamsize)n);
}

}  // namespace weights_detail

// Baut aus den gespeicherten Merge-Regeln denselben Tokenizer wie beim Training.
inline BPETokenizer rebuild_bpe(const std::vector<std::pair<std::string, std::string>>& merges) {
    BPETokenizer tok;
    for (int i = 0; i < 256; ++i) tok.add_token(std::string(1, (char)i));
    for (const auto& m : merges) tok.add_merge(m.first, m.second);   // Token a+b anlegen + Merge merken
    return tok;
}

inline void save_weights_stream(std::ostream& f, const SkullWeights& w) {
    using namespace weights_detail;
    if (w.is_transformer) {
        f.write("SKULT", 5);
        put_u32(f, 1);   // Format-Version
        put_u64(f, (uint64_t)w.tcfg.vocab);
        put_u64(f, (uint64_t)w.tcfg.dim);
        put_u64(f, (uint64_t)w.tcfg.context);
        put_u64(f, (uint64_t)w.tcfg.heads);
        put_u64(f, (uint64_t)w.tcfg.layers);
        f.write(reinterpret_cast<const char*>(w.tparams.data()),
                (std::streamsize)(w.tparams.size() * sizeof(double)));
    } else {
        f.write("SKULL", 5);
        put_u64(f, (uint64_t)w.dim);
        put_u64(f, (uint64_t)w.vocab);
        f.write(reinterpret_cast<const char*>(w.W_embed.data()),
                (std::streamsize)(w.W_embed.size() * sizeof(double)));
        f.write(reinterpret_cast<const char*>(w.W_hidden.data()),
                (std::streamsize)(w.W_hidden.size() * sizeof(double)));
        f.write(reinterpret_cast<const char*>(w.W_out.data()),
                (std::streamsize)(w.W_out.size() * sizeof(double)));
    }

    if (w.has_bpe) {
        f.write("TOKB", 4);
        put_u32(f, (uint32_t)w.bpe.merges.size());
        for (const auto& m : w.bpe.merges) {
            put_str(f, m.first);
            put_str(f, m.second);
        }
    }
}

inline void save_weights(const std::string& path, const SkullWeights& w) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Gewichte konnten nicht geschrieben werden: " + path);
    save_weights_stream(f, w);
    f.flush();
    if (!f)
        throw std::runtime_error("Fehler beim Schreiben der Gewichte: " + path);
}

// Liest den optionalen TOKB-Abschnitt (BPE-Merges) ab der aktuellen Position, falls Daten uebrig sind.
inline void read_tokenizer_section(std::istream& f, uint64_t file_size, uint64_t expected,
                                   SkullWeights& w, uint64_t vocab, const std::string& path) {
    using namespace weights_detail;
    if (file_size <= expected) return;
    char tm[4] = {};
    if (!f.read(tm, 4) || std::memcmp(tm, "TOKB", 4) != 0)
        throw std::runtime_error("Gewichte-Datei: unbekannte Daten nach den Gewichten: " + path);
    uint32_t n = 0;
    if (!get_u32(f, n) || n > MAX_MERGES)
        throw std::runtime_error("Gewichte-Datei: Tokenizer-Abschnitt beschaedigt: " + path);
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        std::string a, b;
        if (!get_str(f, a) || !get_str(f, b))
            throw std::runtime_error("Gewichte-Datei: Tokenizer-Abschnitt abgeschnitten: " + path);
        merges.emplace_back(std::move(a), std::move(b));
    }
    w.bpe     = rebuild_bpe(merges);
    w.has_bpe = true;
    if ((uint64_t)w.bpe.vocab_size() != vocab)
        throw std::runtime_error(
            "Gewichte-Datei: Vokabulargroesse passt nicht zum Tokenizer (" +
            std::to_string(vocab) + " vs " + std::to_string(w.bpe.vocab_size()) + ")");
}

// Liest Gewichte aus einem Stream (Datei oder Speicherabbild); `path` dient nur den Fehlermeldungen.
inline SkullWeights load_weights_stream(std::istream& f, const std::string& path) {
    using namespace weights_detail;
    f.seekg(0, std::ios::end);
    const uint64_t file_size = (uint64_t)f.tellg();
    f.seekg(0, std::ios::beg);

    char magic[5] = {};
    if (!f.read(magic, 5))
        throw std::runtime_error("Ungueltige Gewichte-Datei (zu kurz, Magic 'SKULL'/'SKULT' fehlt): " + path);
    const bool is_bigram      = std::memcmp(magic, "SKULL", 5) == 0;
    const bool is_transformer = std::memcmp(magic, "SKULT", 5) == 0;
    if (!is_bigram && !is_transformer)
        throw std::runtime_error("Ungueltige Gewichte-Datei (Magic 'SKULL'/'SKULT' fehlt): " + path);

    SkullWeights w;

    if (is_transformer) {
        uint32_t version = 0;
        uint64_t vocab = 0, dim = 0, context = 0, heads = 0, layers = 0;
        if (!get_u32(f, version) || !get_u64(f, vocab) || !get_u64(f, dim) ||
            !get_u64(f, context) || !get_u64(f, heads) || !get_u64(f, layers))
            throw std::runtime_error("Gewichte-Datei zu kurz (Header unvollstaendig): " + path);
        if (version != 1)
            throw std::runtime_error("Gewichte-Datei: unbekannte Format-Version " +
                                     std::to_string(version) + " (diese Skull-Version kennt 1): " + path);
        if (vocab == 0 || dim == 0 || context == 0 || heads == 0 || layers == 0 ||
            vocab > MAX_VOCAB || dim > MAX_DIM || context > MAX_CONTEXT || layers > MAX_LAYERS ||
            heads > dim || dim % heads != 0)
            throw std::runtime_error(
                "Gewichte-Datei beschaedigt: unplausible Groessen vocab=" + std::to_string(vocab) +
                ", dim=" + std::to_string(dim) + ", context=" + std::to_string(context) +
                ", heads=" + std::to_string(heads) + ", layers=" + std::to_string(layers));
        TransformerConfig tc;
        tc.vocab = (size_t)vocab; tc.dim = (size_t)dim; tc.context = (size_t)context;
        tc.heads = (size_t)heads; tc.layers = (size_t)layers;
        const uint64_t n_params = (uint64_t)tc.param_count();
        const uint64_t expected = 5 + 4 + 5 * 8 + n_params * sizeof(double);
        if (file_size < expected)
            throw std::runtime_error(
                "Gewichte-Datei abgeschnitten oder beschaedigt: erwartet mindestens " +
                std::to_string(expected) + " Bytes, gefunden " + std::to_string(file_size));
        w.is_transformer = true;
        w.tcfg   = tc;
        w.dim    = tc.dim;
        w.vocab  = tc.vocab;
        w.tparams.resize((size_t)n_params);
        if (!f.read(reinterpret_cast<char*>(w.tparams.data()), (std::streamsize)(n_params * sizeof(double))))
            throw std::runtime_error("Lesefehler in Gewichte-Datei: " + path);
        read_tokenizer_section(f, file_size, expected, w, vocab, path);
        return w;
    }

    uint64_t dim = 0, vocab = 0;
    if (!get_u64(f, dim) || !get_u64(f, vocab))
        throw std::runtime_error("Gewichte-Datei zu kurz (Header unvollstaendig): " + path);
    if (dim == 0 || vocab == 0 || dim > MAX_DIM || vocab > MAX_VOCAB)
        throw std::runtime_error(
            "Gewichte-Datei beschaedigt: unplausible Groessen dim=" + std::to_string(dim) +
            ", vocab=" + std::to_string(vocab));

    const uint64_t n_embed  = vocab * dim;
    const uint64_t n_hidden = dim * dim;
    const uint64_t n_out    = dim * vocab;
    const uint64_t expected = 5 + 16 + (n_embed + n_hidden + n_out) * sizeof(double);
    if (file_size < expected)
        throw std::runtime_error(
            "Gewichte-Datei abgeschnitten oder beschaedigt: erwartet mindestens " +
            std::to_string(expected) + " Bytes, gefunden " + std::to_string(file_size));

    w.dim   = (size_t)dim;
    w.vocab = (size_t)vocab;
    w.W_embed.resize((size_t)n_embed);
    w.W_hidden.resize((size_t)n_hidden);
    w.W_out.resize((size_t)n_out);
    f.read(reinterpret_cast<char*>(w.W_embed.data()),  (std::streamsize)(n_embed  * sizeof(double)));
    f.read(reinterpret_cast<char*>(w.W_hidden.data()), (std::streamsize)(n_hidden * sizeof(double)));
    f.read(reinterpret_cast<char*>(w.W_out.data()),    (std::streamsize)(n_out    * sizeof(double)));
    if (!f)
        throw std::runtime_error("Lesefehler in Gewichte-Datei: " + path);

    read_tokenizer_section(f, file_size, expected, w, vocab, path);
    return w;
}

inline SkullWeights load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            "Gewichte nicht gefunden: " + path + " (zuerst trainieren: train Modell { ... })");
    return load_weights_stream(f, path);
}
