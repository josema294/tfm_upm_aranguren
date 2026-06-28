#!/usr/bin/env python3
import json
import os
import sys
import hashlib
from pathlib import Path

#veriuficador de descargas para comprobar que los hashes coincidenc de lo descargado.

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
        sys.exit(1)

    tipo_detector = os.environ.get("DETECTOR_TYPE", "hybrid")
    print(f"Comprobando los artefactos requeridos para el modo DETECTOR_TYPE={tipo_detector}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifiesto = json.load(f)

    todo_correcto = True

    for ruta_relativa, info in manifiesto.items():
        if tipo_detector not in info.get("necesario_para", []):
            continue

        ruta_destino = ROOT_DIR / ruta_relativa
        if not ruta_destino.exists():
            print(f"❌ Falta un archivo: {ruta_relativa} (Es obligatorio para el modo {tipo_detector})")
            todo_correcto = False
            continue

        if "<" in info["sha256"] and ">" in info["sha256"]:
            print(f"⚠️ Aviso: El hash del manifiesto es un placeholder provisional. No es posible validar {ruta_relativa}.")
            todo_correcto = False
            continue

        if "tamano_bytes" in info and ruta_destino.stat().st_size != int(info["tamano_bytes"]):
            print(f"❌ Tamaño inesperado: {ruta_relativa}")
            todo_correcto = False
            continue

        hash_actual = calculate_sha256(ruta_destino)
        if hash_actual != info["sha256"]:
            print(f"❌ Archivo corrupto: {ruta_relativa}\n   - Hash esperado: {info['sha256']}\n   - Hash actual:   {hash_actual}")
            todo_correcto = False
        else:
            print(f"✅ Validación correcta: {ruta_relativa}")

    if todo_correcto:
        print("\n✨ Verificación finalizada con éxito. Todos los modelos están listos.")
        sys.exit(0)
    else:
        print("\n❌ Ha fallado la verificación. Revisa el registro superior porque faltan pesos o están corruptos.")
        sys.exit(1)

if __name__ == "__main__":
    if not MANIFEST_PATH.exists() and (ROOT_DIR / MANIFEST_PATH).exists():
        os.chdir(ROOT_DIR)
    main()
