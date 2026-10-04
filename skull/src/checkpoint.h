#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include "weights.h"

// ============================================================
//  SKULL CHECKPOINT  —  kompletter Trainingszustand zum Pausieren und Fortsetzen
//
//  Eine .weights-Datei enthaelt nur das fertige Modell. Zum FORTSETZEN eines Trainings
//  reicht das nicht: Adam braucht seine Momente, der Zufallsgenerator seinen Stand,
//  der Lernraten-Plan seine Position, der Rahmen die bisherige Bestleistung. Ein
//  Checkpoint enthaelt all das, damit ein fortgesetztes Training BITGLEICH dasselbe
//  Ergebnis liefert wie ein durchgehender Lauf (tests/traincheck.cpp).
//
//  Dateiformat (little-endian):
//    "SKULC"            5 Byte Magic
//    uint32  version (= 1)
//    uint32  kind       0 = Bigram, 1 = Transformer
//    uint64  data_hash  Pruefsumme der Trainings- und Validierungs-Token
//    int32   epoch      zuletzt abgeschlossene Epoche
//    double  best_val, last_loss;  int32 best_epoch, bad_epochs;  uint8 has_best
//    [wenn has_best: blob  Gewichte des besten Validierungs-Modells (SKULL/SKULT-Format)]
//    blob    modellspezifischer Zustand
//  blob = uint64 Laenge + Bytes. Beim Laden wird jede Groesse gegen die erwarteten Werte
//  geprueft, bevor Speicher angelegt wird.
// ============================================================

namespace ckpt {

constexpr uint32_t KIND_BIGRAM      = 0;
constexpr uint32_t KIND_TRANSFORMER = 1;
constexpr uint64_t MAX_BLOB = 1ull << 40;

inline void put_u8(std::ostream& o, uint8_t v) { o.write(reinterpret_cast<const char*>(&v), 1); }
inline bool get_u8(std::istream& in, uint8_t& v) { return (bool)in.read(reinterpret_cast<char*>(&v), 1); }
inline void put_i32(std::ostream& o, int32_t v) { weights_detail::put_u32(o, (uint32_t)v); }
inline bool get_i32(std::istream& in, int32_t& v) {
    uint32_t u = 0;
    if (!weights_detail::get_u32(in, u)) return false;
    v = (int32_t)u;
    return true;
}
inline void put_f64(std::ostream& o, double v) {
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    weights_detail::put_u64(o, u);
}
inline bool get_f64(std::istream& in, double& v) {
    uint64_t u = 0;
    if (!weights_detail::get_u64(in, u)) return false;
    std::memcpy(&v, &u, sizeof v);
    return true;
}

inline void put_blob(std::ostream& o, const std::string& b) {
    weights_detail::put_u64(o, (uint64_t)b.size());
    o.write(b.data(), (std::streamsize)b.size());
}
// Liest einen Blob; max_len begrenzt die erlaubte Groesse (Schutz vor beschaedigten Dateien).
inline bool get_blob(std::istream& in, std::string& b, uint64_t max_len) {
    uint64_t n = 0;
    if (!weights_detail::get_u64(in, n) || n > max_len) return false;
    b.resize((size_t)n);
    return n == 0 || (bool)in.read(&b[0], (std::streamsize)n);
}

// Vektor von doubles: Anzahl + Rohdaten. Die erwartete Anzahl wird beim Lesen erzwungen.
inline void put_doubles(std::ostream& o, const std::vector<double>& v) {
    weights_detail::put_u64(o, (uint64_t)v.size());
    o.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(double)));
}
inline bool get_doubles(std::istream& in, std::vector<double>& v, size_t expected) {
    uint64_t n = 0;
    if (!weights_detail::get_u64(in, n) || n != (uint64_t)expected) return false;
    v.resize(expected);
    return expected == 0 || (bool)in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(expected * sizeof(double)));
}

// FNV-1a (64 Bit) ueber Token-IDs: erkennt, dass sich Daten, Tokenisierung oder Aufteilung geaendert haben.
inline uint64_t hash_ids(const std::vector<int>& train, const std::vector<int>& val) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t x) {
        for (int i = 0; i < 8; ++i) { h ^= (x >> (8 * i)) & 0xFF; h *= 1099511628211ull; }
    };
    mix((uint64_t)train.size());
    for (int id : train) mix((uint64_t)(uint32_t)id);
    mix((uint64_t)val.size() + 0x9E3779B97F4A7C15ull);
    for (int id : val) mix((uint64_t)(uint32_t)id);
    return h;
}

// Zustand des Trainingsrahmens (siehe run_training in trainer.h)
struct DriverState {
    int          epoch      = 0;       // zuletzt abgeschlossene Epoche
    double       best_val   = 0.0;
    double       last_loss  = 0.0;
    int          best_epoch = 0;
    int          bad_epochs = 0;
    bool         has_best   = false;
    SkullWeights best;
};

inline std::string weights_to_string(const SkullWeights& w) {
    std::ostringstream os(std::ios::binary);
    save_weights_stream(os, w);
    return os.str();
}
inline SkullWeights weights_from_string(const std::string& bytes, const std::string& label) {
    std::istringstream is(bytes, std::ios::binary);
    return load_weights_stream(is, label);
}

// Schreibt atomar: erst in <path>.tmp, dann umbenennen. Ein abgebrochener Schreibvorgang
// zerstoert so nie den letzten guten Checkpoint.
inline void write_checkpoint(const std::string& path, uint32_t kind, uint64_t data_hash,
                             const DriverState& st, const std::string& model_state) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Checkpoint konnte nicht geschrieben werden: " + path);
        f.write("SKULC", 5);
        weights_detail::put_u32(f, 1);
        weights_detail::put_u32(f, kind);
        weights_detail::put_u64(f, data_hash);
        put_i32(f, st.epoch);
        put_f64(f, st.best_val);
        put_f64(f, st.last_loss);
        put_i32(f, st.best_epoch);
        put_i32(f, st.bad_epochs);
        put_u8(f, st.has_best ? 1 : 0);
        if (st.has_best) put_blob(f, weights_to_string(st.best));
        put_blob(f, model_state);
        f.flush();
        if (!f) throw std::runtime_error("Fehler beim Schreiben des Checkpoints: " + path);
    }
    std::remove(path.c_str());   // Windows: rename ueberschreibt nicht
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("Checkpoint konnte nicht umbenannt werden: " + tmp + " -> " + path);
}

// Liest einen Checkpoint; gibt den Zustand des Rahmens zurueck, den modellspezifischen Blob in model_state.
inline DriverState read_checkpoint(const std::string& path, uint32_t expected_kind,
                                   uint64_t expected_hash, std::string& model_state) {
    using namespace weights_detail;
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Checkpoint nicht gefunden: " + path);
    auto bad = [&](const char* why) { return std::runtime_error(std::string("Checkpoint ") + why + ": " + path); };

    char magic[5] = {};
    if (!f.read(magic, 5) || std::memcmp(magic, "SKULC", 5) != 0) throw bad("ungueltig (Magic 'SKULC' fehlt)");
    uint32_t version = 0, kind = 0;
    uint64_t hash = 0;
    if (!get_u32(f, version) || !get_u32(f, kind) || !get_u64(f, hash)) throw bad("abgeschnitten");
    if (version != 1) throw std::runtime_error("Checkpoint: unbekannte Version " + std::to_string(version) + ": " + path);
    if (kind != expected_kind)
        throw std::runtime_error(std::string("Checkpoint gehoert zu einem ") +
                                 (kind == KIND_TRANSFORMER ? "Transformer" : "Bigram") +
                                 "-Modell, trainiert wird aber ein " +
                                 (expected_kind == KIND_TRANSFORMER ? "Transformer" : "Bigram") + "-Modell: " + path);
    if (hash != expected_hash)
        throw std::runtime_error("Checkpoint passt nicht zu den aktuellen Daten (andere Datei, Tokenisierung, "
                                 "val oder bpe?): " + path);

    DriverState st;
    uint8_t has_best = 0;
    if (!get_i32(f, st.epoch) || !get_f64(f, st.best_val) || !get_f64(f, st.last_loss) ||
        !get_i32(f, st.best_epoch) || !get_i32(f, st.bad_epochs) || !get_u8(f, has_best))
        throw bad("abgeschnitten");
    if (st.epoch < 0 || st.best_epoch < 0 || st.bad_epochs < 0) throw bad("beschaedigt");
    st.has_best = has_best != 0;
    if (st.has_best) {
        std::string blob;
        if (!get_blob(f, blob, MAX_BLOB)) throw bad("abgeschnitten (Bestes Modell)");
        st.best = weights_from_string(blob, path);
    }
    if (!get_blob(f, model_state, MAX_BLOB)) throw bad("abgeschnitten (Modellzustand)");
    return st;
}

}  // namespace ckpt
