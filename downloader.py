import os
import requests
from urllib.parse import urljoin
import re

def download_files_from_directory(url, download_dir="downloaded_files"):
    """
    Скачивает все файлы из директории на сайте PDS
    """
    # Создаем директорию для скачивания, если она не существует
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Получаем содержимое страницы
    try:
        print("Подключаемся к сайту...")
        response = requests.get(url)
        response.raise_for_status()
        print("Подключение успешно!")
    except requests.RequestException as e:
        print(f"Ошибка при подключении к сайту: {e}")
        return
    
    # Парсим ссылки на файлы с помощью регулярных выражений
    file_links = []
    
    # Ищем все ссылки в формате HREF="filename"
    pattern = r'HREF="([^"]*\.(?:dat|xml))"'
    matches = re.findall(pattern, response.text)
    
    if matches:
        file_links = matches
        print(f"Найдено файлов: {len(file_links)}")
    else:
        print("Файлы не найдены в HTML")
        return
    
    # Скачиваем каждый файл
    successful_downloads = 0
    
    for filepath in file_links:
        # Извлекаем только имя файла из пути
        filename = os.path.basename(filepath)
        file_url = urljoin(url, filepath)
        local_path = os.path.join(download_dir, filename)
        
        print(f"Скачивается: {filename}")
        
        try:
            # Скачиваем файл
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()
            
            # Сохраняем файл
            with open(local_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(local_path)
            print(f"✓ Успешно: {filename} ({file_size} байт)")
            successful_downloads += 1
            
        except requests.RequestException as e:
            print(f"✗ Ошибка при скачивании {filename}: {e}")
        except Exception as e:
            print(f"✗ Ошибка при сохранении {filename}: {e}")
    
    print(f"\nСкачивание завершено!")
    print(f"Успешно скачано: {successful_downloads}/{len(file_links)} файлов")
    print(f"Файлы сохранены в папку: {download_dir}")

if __name__ == "__main__":
    # URL сайта
    url = "https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/data-odf/2015/"
    
    # Запускаем скачивание
    download_files_from_directory(url)