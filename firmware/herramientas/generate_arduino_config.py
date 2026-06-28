#!/usr/bin/env python3
import argparse
import pathlib


def parse_env(ruta_archivo):
    valores = {}
    for linea_cruda in ruta_archivo.read_text().splitlines():
        linea = linea_cruda.strip()
        if not linea or linea.startswith("#"):
            continue
        if "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        valores[clave.strip()] = valor.strip().strip('"').strip("'")
    return valores


def cpp_string(valor):
    return valor.replace("\\", "\\\\").replace('"', '\\"')


def main():
    parser = argparse.ArgumentParser(description="Generador del archivo config.h para Arduino a partir del .env local.")
    parser.add_argument("--env", default=".env", help="Ruta al archivo .env.")
    parser.add_argument(
        "--output",
        default="config.h",
        help="Ruta donde se generará el archivo config.h.",
    )
    args = parser.parse_args()

    ruta_env = pathlib.Path(args.env)
    ruta_salida = pathlib.Path(args.output)
    valores = parse_env(ruta_env)

    requeridos = ["WIFI_SSID", "WIFI_PASSWORD", "DESTINATION_IP"]
    faltantes = [clave for clave in requeridos if clave not in valores or not valores[clave]]
    if faltantes:
        raise SystemExit(f"Error: Faltan variables obligatorias en el archivo .env: {', '.join(faltantes)}")

    puerto_destino = int(valores.get("DESTINATION_PORT", "5005"))
    canal_wifi = int(valores.get("WIFI_CHANNEL", "0"))

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(
        "\n".join(
            [
                "#pragma once",
                "",
                f'const char *WIFI_SSID = "{cpp_string(valores["WIFI_SSID"])}";',
                f'const char *WIFI_PASSWORD = "{cpp_string(valores["WIFI_PASSWORD"])}";',
                f'const char *DESTINATION_IP = "{cpp_string(valores["DESTINATION_IP"])}";',
                f"const uint16_t DESTINATION_PORT = {puerto_destino};",
                "",
                "// Establecer a 0 para permitir que el ESP32 elija el canal automáticamente.",
                f"const int WIFI_CHANNEL = {canal_wifi};",
                "",
            ]
        )
    )
    print(f"Archivo de configuración generado correctamente en {ruta_salida}")


if __name__ == "__main__":
    main()
