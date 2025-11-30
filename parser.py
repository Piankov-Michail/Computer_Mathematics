# parser.py — robust, pdr-free version for MESSENGER ODF
import struct
import pandas as pd
from datetime import datetime, timezone
import os, glob

def parse_messenger_odf(label_path: str):
    import xml.etree.ElementTree as ET

    tree = ET.parse(label_path)
    root = tree.getroot()

    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    orbit_table = None
    for tbl in root.findall(".//pds:Table_Binary", ns):
        name_elem = tbl.find("pds:name", ns)
        if name_elem is not None and name_elem.text.strip() == "ODF Orbit Data Group Data":
            orbit_table = tbl
            break

    if orbit_table is None:
        raise ValueError("'ODF Orbit Data Group Data' table not found in XML.")

    offset_elem = orbit_table.find("pds:offset", ns)
    records_elem = orbit_table.find("pds:records", ns)

    offset = int(offset_elem.text)
    n_records = int(records_elem.text)
    print(f"Found 'ODF Orbit Data Group Data': offset={offset}, records={n_records}")

    dat_path = label_path.replace(".xml", ".dat")
    with open(dat_path, "rb") as f:
        f.seek(offset)
        raw = f.read(36 * n_records)

    if len(raw) != 36 * n_records:
        raise ValueError(f"Data truncated: expected {36 * n_records} bytes, got {len(raw)}")

    records = []
    ref_epoch = datetime(1950, 1, 1, tzinfo=timezone.utc)

    for i in range(n_records):
        start = i * 36
        rec = raw[start:start+36]

        t_int = struct.unpack('>I', rec[0:4])[0]

        packed_23 = struct.unpack('>I', rec[4:8])[0]
        t_frac_ms = (packed_23 >> 22) & 0x3FF

        obs_int = struct.unpack('>i', rec[8:12])[0]
        obs_frac = struct.unpack('>i', rec[12:16])[0]
        doppler = obs_int + obs_frac * 1e-9

        packed_6_14 = struct.unpack('>I', rec[16:20])[0]
        station_id = (packed_6_14 >> 22) & 0x7F
        data_type = (packed_6_14 >> 7) & 0x3F
        valid = (packed_6_14 & 0x1) == 0

        total_sec = t_int + t_frac_ms / 1000.0
        time_utc = ref_epoch + pd.Timedelta(seconds=total_sec)

        records.append({
            "time_utc": time_utc,
            "doppler_hz": doppler,
            "station_id": station_id,
            "data_type": data_type,
            "valid": valid
        })

    df = pd.DataFrame(records)
    doppler_df = df[(df["valid"]) & (df["data_type"].isin([11, 12, 13]))].reset_index(drop=True)

    print(f"✅ Parsed {len(doppler_df)} valid Doppler records.")
    return doppler_df


if __name__ == "__main__":
    # Папки для входных и выходных файлов
    input_dir = "downloaded_files"
    output_dir = "data"
    
    # Создаем папку для выходных данных, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана папка для выходных данных: {output_dir}")
    
    # Ищем XML файлы в папке downloaded_files
    file_pattern = os.path.join(input_dir, "mess_rs_*.xml")
    xml_files = glob.glob(file_pattern)
    
    if not xml_files:
        print(f"Файлы, соответствующие шаблону '{file_pattern}', не найдены.")
        print("Убедитесь, что:")
        print(f"1. Папка '{input_dir}' существует")
        print(f"2. В папке '{input_dir}' есть файлы mess_rs_*.xml")
        print(f"3. Для каждого XML файла есть соответствующий DAT файл")
    else:
        print(f"Найдено {len(xml_files)} файлов для обработки в папке '{input_dir}'")

        for xml_file in xml_files:
            try:
                print(f"\nОбрабатывается файл: {os.path.basename(xml_file)}")
                
                # Парсим данные
                df = parse_messenger_odf(xml_file)
                
                # Формируем имя выходного CSV файла
                base_name = os.path.splitext(os.path.basename(xml_file))[0]
                csv_file = os.path.join(output_dir, base_name + ".csv")
                
                # Сохраняем в CSV
                df.to_csv(csv_file, index=False)
                
                print(f"✅ Файл обработан и сохранен в: {csv_file}")
                print(f"   Количество записей: {len(df)}")
                
            except Exception as e:
                print(f"❌ Ошибка при обработке файла {xml_file}: {e}")
                continue

        print(f"\nОбработка всех файлов завершена.")
        print(f"CSV файлы сохранены в папку: {output_dir}")