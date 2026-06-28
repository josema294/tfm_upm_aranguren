#!/usr/bin/env python3
import json
import os
import hashlib
import sys
import urllib.request
from pathlib import Path

MANIFEST_PATH = Path("artefactos/manifiesto.json")
ROOT_DIR = Path(__file__).parent.parent

def calculate_sha256(ruta_archivo):
    hash_sha256 = hashlib.sha256()
    with open(ruta_archivo, "rb") as f:
        for bloque in iter(lambda: f.read(4096), b""):
            hash_sha256.update(bloque)
    return hash_sha256.hexdigest()

def main():
    if not MANIFEST_PATH.exists():
        print(f"Error: No he podido encontrar el archivo de manifiesto en {MANIFEST_PATH}")
        return 1

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifiesto = json.load(f)

    todo_correcto = True

    for ruta_relativa, info in manifiesto.items():
        ruta_destino = ROOT_DIR / ruta_relativa
        
        if "<" in info["url"] and ">" in info["url"]:
            print(f"Error: la URL de {ruta_relativa} no esta configurada: {info['url']}")
            todo_correcto = False
            continue

        if "<" in info["sha256"] and ">" in info["sha256"]:
            print(f"Error: el SHA256 de {ruta_relativa} no esta configurado.")
            todo_correcto = False
            continue

        if ruta_destino.exists():
            if "tamano_bytes" in info and ruta_destino.stat().st_size != int(info["tamano_bytes"]):
                print(f"Atención: El tamaño no coincide para {ruta_relativa}, así que vamos a volver a descargarlo.")
            else:
                
                hash_actual = calculate_sha256(ruta_destino)
                if hash_actual == info["sha256"]:
                    print(f"El archivo {ruta_relativa} ya existe y su hash es correcto.")
                    continue
                else:
                    print(f"Atención: El hash no coincide para {ruta_relativa}, así que vamos a volver a descargarlo.")

        print(f"Procediendo a descargar {ruta_relativa}...")
        ruta_destino.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            urllib.request.urlretrieve(info["url"], ruta_destino)
            
            if "<" not in info["sha256"]:
                nuevo_hash = calculate_sha256(ruta_destino)
                if nuevo_hash != info["sha256"]:
                    print(f"Error de integridad: El hash del archivo descargado ({ruta_relativa}) no se corresponde con el esperado.")
                    ruta_destino.unlink()
                    todo_correcto = False
                else:
                    print(f"Todo en orden, el archivo {ruta_relativa} se ha descargado y validado correctamente.")
                
        except Exception as e:
            print(f"Ha ocurrido un error al intentar descargar {ruta_relativa}: {e}")
            todo_correcto = False

    return 0 if todo_correcto else 1

if __name__ == "__main__":
    if not MANIFEST_PATH.exists() and (ROOT_DIR / MANIFEST_PATH).exists():
        os.chdir(ROOT_DIR)
    sys.exit(main())
