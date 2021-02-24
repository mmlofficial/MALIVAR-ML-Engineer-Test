# MALIVAR-ML-Engineer-Test
Test task for generation and editing artificial faces

## Задача
Задача состоит из двух частей:

1. Генерация несуществующего человека с возможностью редактирования лица.
Скрипт с множеством гиперпараметров или окно с элементами управления.
2. Скрипт, который генерирует изображения ранее сгенерированного человека (головы) с комбинацией различных вращений и выражений лица.

### В данной работе использовались
* Nvidea GeForce 960 4Gb VRAM
* Python 3.6
* CUDA 10.1
* cuDNN v7.6.5
## 1 часть задания
Для выполнения первой части задания был найден и изучен open-source проект __TL-GAN: transparent latent-space GAN__ (https://github.com/SummitKwan/transparent_latent_gan)

### Установка и запуск
1. Создание виртуального окружения для данной части задания  
    ```
    > python -m venv venvtlgan

    > venvtlgan\Scripts\activate
    ```
2. Установка зависимостей и запуск
    1. ```cd``` (в корневую директорию проекта).  
    
    ```
    > cd transparent_latent_gan  

    > pip install -r requirements.txt
    ```
    2. Вручную загрузите предварительно обученную модель pg-GAN из [dropbox](https://www.dropbox.com/sh/y1ryg8iq1erfcsr/AAB--PO5qAapwp8ILcgxE2I6a?dl=0).
    3. Распакуйте загруженные файлы и поместите их в каталог проекта в следующем формате
    ```
    root
        asset_model
            karras2018iclr-celebahq-1024x1024.pkl
            cnn_face_attr_celeba
                model_20180927_032934.h5
        asset_results
            pg_gan_celeba_feature_direction_40
                feature_direction_20181002_044444.pkl
        src
            ...
        static
            ...
        ...
    ```
    4. Запустите jupyter ноутбук (потребуется загрузить соответствующую библиотеку) из ```./src/notebooks/tl_gan_ipywidgets_gui.ipynb``` 
    5. В GUI можно настроить лицо несуществующего человека с выбранными параметрами и сохранить в папке для использования изображения во второй части задания.

3. Пример работы  
      
    ![Alt text](./gifs/online_demo_run_fast_01.gif?raw=true "Title")  
    Данный проект был расширен только путем добавления кнопки _Save image_ для дальнейшего использования.

## 2 часть задания
Во второй части задания использовался open-source проект __CONFIG: Controllable Neural Face Image Generation__ (https://arxiv.org/pdf/2005.02671.pdf) (https://github.com/microsoft/ConfigNet)

### Установка и запуск
1. Создание виртуального окружения для 2-ой части задания  
Перед данным действием требуется покинуть виртуальное окружение 1-ого задания.
    ```
    > python -m venv venvconfignet

    > venvconfignet\Scripts\activate
    ```
2. Установка зависимостей и запуск
    1. ```cd``` (в корневую директорию проекта).  
    