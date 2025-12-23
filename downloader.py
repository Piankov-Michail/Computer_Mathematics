import os
import requests
from urllib.parse import urljoin
import re

def download_files_from_directory(url, download_dir="downloaded_files"):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    try:
        print("Подключаемся к сайту...")
        response = requests.get(url)
        response.raise_for_status()
        print("Подключение успешно!")
    except requests.RequestException as e:
        print(f"Ошибка при подключении к сайту: {e}")
        return
    
    file_links = []
    
    pattern = r'HREF="([^"]*\.(?:dat|xml))"'
    matches = re.findall(pattern, response.text)
    
    if matches:
        file_links = matches
        print(f"Найдено файлов: {len(file_links)}")
    else:
        print("Файлы не найдены в HTML")
        return
    
    successful_downloads = 0
    
    for filepath in file_links:
        filename = os.path.basename(filepath)
        file_url = urljoin(url, filepath)
        local_path = os.path.join(download_dir, filename)
        
        print(f"Скачивается: {filename}")
        
        try:
            file_response = requests.get(file_url, stream=True)
            file_response.raise_for_status()

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
    url = "https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/data-odf/2014/"
    
    download_files_from_directory(url)