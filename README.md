# QuasarNet 🌠

**Astro Spectrum Classifier**  
`CNN-Transformer` model to classify spectra (Quasar/Star/Black Hole) from FITS files.

```bash
# Install & Run
pip install torch astropy flask
wget bit.ly/quasarnet-weights -O model.pth
python app.py  # API on :5000
Usage:

python
Copy
import requests
r = requests.post("http://localhost:5000/classify", 
                 files={"file": open("spec.fits","rb")})
print(r.json())  # {'prediction':'Quasar', 'confidence':0.95}
Features:

🚀 Hybrid AI model

🔭 SDSS/DESI FITS support

⚡ <1s inference

Tech: Python • PyTorch • Flask

MIT License • Avika

Copy

Key points:
1. Ultra-compact (fits GitHub mobile preview)
2. Self-contained installation/usage
3. Essential badges via emoji
4. Working code snippets
5. Minimal dependencies
6. License + credit at bottom
