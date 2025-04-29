# Instruksi
- Scrap data dari masing - masing outlet sebanyak 2000 data.
- Apabila Data yang di scrap berdasarkan events pada main.py tidak mencapai 2000, scrap data dari kategori atau judul apapun dengan     event sebagai others
- Masing - masing event datanya tidak harus sama atau equal (misal covid 80 data, trump 90 data)
- Format output berbentuk parquet
- Masing masing outlet memiliki outputnya sendiri, yang nanti di merge (guardian.parquet, fox.parquet, etc)

## Data yang belum di scrap:

1. Jacobin
1. Fox
3. Breitbart

# Cara run program
1. install dependecies
```
pip install -r requrements.txt
```
2. python main.py (scrap semua outlet), python main.py --source bbc --debug (scrap semua events pada satu outlet), python main.py --source bbc --event 0 --debug (test 1 event pada 1 outlet)

# Run command
Powershell
```
cd backend; uvicorn main:app --host 0.0.0.0 --port 8000
```