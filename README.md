# Dataset Audio Mixed - Sistem Analisis Kualitas Audio

Sistem Python untuk memproses, menganalisis, dan mengevaluasi kualitas audio dengan berbagai tingkat noise menggunakan metrik seperti PESQ, STOI, dan MSE.

## Instalasi

### Persyaratan Sistem

- Python 3.8 atau lebih tinggi
- FFmpeg (untuk konversi audio)
- Git (opsional)

### Langkah Instalasi

1. **Clone atau Download Repository**

   ```bash
   git clone https://github.com/MochNad/Dataset-DTLN.git
   cd Dataset-DTLN
   ```

2. **Buat Virtual Environment (Disarankan)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Struktur Folder

```
Dataset-DTLN/
├── log/                          # File log otomatis
├── formats/                      # Hasil formatting
│   ├── cleans/                   # Audio bersih terformat
│   └── noises/                   # Audio noise terformat
├── generates/                    # Hasil generasi
│   ├── datasets/                 # Dataset yang dihasilkan
│   └── experiments/              # Struktur eksperimen
├── analyses/                     # Hasil analisis
│   ├── metrics/                  # Metrik per file
│   │   ├── en/                   # Data bahasa Inggris
│   │   └── id/                   # Data bahasa Indonesia
│   └── visuals/                  # Visualisasi spectrogram
├── evaluates/                    # Hasil evaluasi
│   ├── metrics.csv               # Ringkasan metrik
│   ├── charts.png                # Grafik metrik
│   └── visuals/                  # Visualisasi evaluasi
└── requirements.txt              # Dependencies Python
```

## Daftar File dan Fungsinya

### 1. formating_clean.py

**Fungsi**: Format dan konversi file audio bersih

- Mengkonversi audio ke format WAV mono 16kHz
- Filter durasi audio 5-6 detik
- Pemrosesan paralel untuk performa optimal

**Penggunaan**:

```bash
python formating_clean.py -i /path/to/audio/folder -o formats/cleans
```

**Argumen**:

- `-i, --input`: Folder input berisi file audio (wajib)
- `-o, --output`: Folder output (default: formats/cleans)
- `-j, --jobs`: Jumlah proses paralel (default: CPU count)
- `-v, --verbose`: Mode verbose

### 2. formating_noise.py

**Fungsi**: Format dan organisir file audio noise

- Ekstrak file ch01 dari folder noise
- Kategorisasi otomatis berdasarkan nama folder
- Konversi ke format standar WAV 16kHz

**Penggunaan**:

```bash
python formating_noise.py -i /path/to/noise/folder -o formats/noises
```

**Argumen**:

- `-i, --input`: Folder root berisi file noise (wajib)
- `-o, --output`: Folder output (default: formats/noises)
- `-v, --verbose`: Mode verbose

**Kategori Noise**:

- D\* → domestic (rumah tangga)
- N\* → nature (alam)
- O\* → office (kantor)
- P\* → public (publik)
- S\* → street (jalan)
- T\* → transportation (transportasi)

### 3. generating_dataset.py

**Fungsi**: Buat dataset campuran audio bersih + noise

- Campur audio bersih dengan noise pada berbagai SNR
- Mendukung bahasa Indonesia dan Inggris
- Generate file dengan variasi SNR -5, 0, 5, 10 dB

**Penggunaan**:

```bash
python generating_dataset.py --clean formats/cleans --noise formats/noises -o generates/datasets
```

**Argumen**:

- `--clean`: Folder audio bersih dengan subfolder en/ dan id/ (wajib)
- `--noise`: Folder noise berkategori (wajib)
- `-o, --output`: Folder output dataset (default: generates/datasets)
- `--max`: Maksimal file per bahasa (default: 30)
- `-v, --verbose`: Mode verbose

### 4. generating_experiment.py

**Fungsi**: Buat struktur folder eksperimen

- Membuat hierarki folder untuk eksperimen
- Tidak menyalin file, hanya struktur folder
- Siap untuk diisi dengan hasil pemrosesan

**Penggunaan**:

```bash
python generating_experiment.py -i generates/datasets -o generates/experiments
```

**Argumen**:

- `-i, --input`: Folder dataset input (wajib)
- `-o, --output`: Folder eksperimen output (default: generates/experiments)
- `--force`: Hapus folder eksisting
- `--dry-run`: Mode preview tanpa eksekusi
- `-v, --verbose`: Mode verbose

### 5. analyzing_metric.py

**Fungsi**: Analisis metrik kualitas audio

- Hitung PESQ, STOI, MSE untuk pasangan audio
- Ekstrak metrik performa dari file CSV
- Simpan hasil per file audio bersih

**Penggunaan**:

```bash
python analyzing_metric.py --dataset generates/datasets --experiment generates/experiments -o analyses/metrics
```

**Argumen**:

- `--dataset`: Folder dataset berisi audio asli (wajib)
- `--experiment`: Folder eksperimen berisi audio hasil (wajib)
- `-o, --output`: Folder output analisis (default: analyses/metrics)
- `-v, --verbose`: Mode verbose

**Metrik yang Dihitung**:

- **PESQ**: Perceptual Evaluation of Speech Quality (1.0-4.5)
- **STOI**: Short-Time Objective Intelligibility (0.0-1.0)
- **MSE**: Mean Squared Error (normalized)
- **Performance**: Buffer, Worklet, Worker, Model timing

### 6. analyzing_visual.py

**Fungsi**: Buat visualisasi spectrogram perbandingan

- Spectrogram audio bersih, noise, campuran, hasil
- Layout 4 panel horizontal
- Colormap plasma untuk visual yang jelas

**Penggunaan**:

```bash
python analyzing_visual.py --dataset generates/datasets --experiment generates/experiments -o analyses/visuals
```

**Argumen**:

- `--dataset`: Folder dataset (wajib)
- `--experiment`: Folder eksperimen (wajib)
- `-o, --output`: Folder output visualisasi (default: analyses/visuals)
- `--dpi`: DPI output gambar (default: 100)
- `-v, --verbose`: Mode verbose

### 7. evaluating_metric.py

**Fungsi**: Evaluasi dan ringkasan metrik

- Gabungkan semua file metrik individual
- Hitung statistik min, max, mean, std
- Generate grafik ringkasan

**Penggunaan**:

```bash
python evaluating_metric.py --metric analyses/metrics -o evaluates/metrics.csv
```

**Argumen**:

- `--metric`: Folder berisi file metrik individual (wajib)
- `-o, --output`: File output CSV ringkasan (default: evaluates/metrics.csv)
- `-v, --verbose`: Mode verbose

### 8. evaluating_visual.py

**Fungsi**: Buat visualisasi evaluasi komprehensif

- Grafik kualitas vs SNR
- Perbandingan kategori noise
- Korelasi metrik kualitas
- Perbandingan bahasa

**Penggunaan**:

```bash
python evaluating_visual.py --metric evaluates/metrics.csv
```

**Atau untuk file individual**:

```bash
python evaluating_visual.py --metric analyses/metrics
```

**Argumen**:

- `--metric`: File CSV ringkasan atau folder metrik individual (wajib)
- `-v, --verbose`: Mode verbose

**Grafik yang Dihasilkan**:

1. Quality vs SNR Line Chart
2. Quality by Noise Category Bar Chart
3. Quality vs Performance Scatter Plot
4. Language Comparison Bar Chart
5. Metrics Correlation Pair Plot

## Workflow Lengkap

### 1. Persiapan Data

```bash
# Format audio bersih
python formating_clean.py -i /path/to/clean/audio -o formats/cleans

# Format audio noise
python formating_noise.py -i /path/to/noise/audio -o formats/noises
```

### 2. Generate Dataset dan Eksperimen

```bash
# Buat dataset campuran
python generating_dataset.py --clean formats/cleans --noise formats/noises --max 30

# Buat struktur eksperimen
python generating_experiment.py -i generates/datasets
```

### 3. Analisis (Setelah Eksperimen Audio Processing)

```bash
# Analisis metrik
python analyzing_metric.py --dataset generates/datasets --experiment generates/experiments

# Analisis visual
python analyzing_visual.py --dataset generates/datasets --experiment generates/experiments
```

### 4. Evaluasi Hasil

```bash
# Evaluasi metrik
python evaluating_metric.py --metric analyses/metrics

# Evaluasi visual
python evaluating_visual.py --metric evaluates/metrics.csv
```

## Format File yang Didukung

### Input Audio

- WAV, MP3, FLAC, M4A, OGG, AAC, WMA

### Output

- **Audio**: WAV 16-bit PCM, 16kHz, Mono
- **Data**: CSV dengan encoding UTF-8
- **Visualisasi**: PNG dengan DPI tinggi

## Logging

Semua script menghasilkan log di folder `log/`:

- `formating_clean.log`
- `formating_noise.log`
- `generating_dataset.log`
- `generating_experiment.log`
- `analyzing_metric.log`
- `analyzing_visual.log`
- `evaluating_metric.log`
- `evaluating_visual.log`

## Tips Penggunaan

1. **Urutan Eksekusi**: Ikuti workflow lengkap dari atas ke bawah
2. **Memory**: Gunakan `-j` untuk membatasi proses paralel jika RAM terbatas
3. **Storage**: Pastikan ruang disk cukup (~1GB per 100 file audio)
4. **Verbose**: Gunakan `-v` untuk debugging dan monitoring detail
5. **Backup**: Backup data original sebelum processing

## Troubleshooting

### Error "No module named"

```bash
pip install -r requirements.txt
```

### Error "Permission denied"

- Windows: Jalankan CMD sebagai Administrator
- Linux/Mac: Gunakan `sudo` atau ubah permission folder

### Error "FFmpeg not found"

- Windows: Download FFmpeg dan tambahkan ke PATH
- Linux: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`

### Audio tidak terbaca

- Pastikan format file didukung
- Cek integritas file audio
- Gunakan mode verbose untuk detail error

## Kontribusi

1. Fork repository
2. Buat branch fitur baru
3. Commit perubahan
4. Push ke branch
5. Buat Pull Request

## Lisensi

[Tentukan lisensi sesuai kebutuhan project]
