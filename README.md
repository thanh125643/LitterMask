# Giới thiệu

Dự án nhằm mục đích sử dụng Mask R-CNN cho việc nhận diện phân vùng rác dựa trên tập dữ liệu TACO có sẵn. Sau đó xây dựng một trang web dùng để phân vùng hình ảnh đó bằng flask

[TACO dataset](https://github.com/pedropro/TACO.git)

[Mask R-CNN](https://github.com/matterport/Mask_RCNN)

# Cài đặt

## Yêu cầu trước khi cài đặt

Visual C++ 2015 Build Tools

Python 3.6, Kereas 2.1.6, Tensorflow 1.12 đều được ghi trong requirement.txt và env.yml

Cần cài đặt [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Các bước cài đặt

Bước 1:Git clone source code về máy.

```bash
git clone https://github.com/thanh125643/LitterMask.git
```

Buớc 2:Khởi tạo môi trường FlaskTaco

```bash
conda create -n FlaskTaco python=3.6

conda activate FlaskTaco
Cho máy có GPU:
conda install -c anaconda tensorflow-gpu==1.12 keras==2.1.6
Cho máy không có GPU(CPU):
conda install -c anaconda tensorflow-mkl==1.12 keras==2.1.6

conda install opencv imgaug==0.4 tqdm jupyter
pip install flask flask-session
```

## Lưu ý

Trường hợp muốn gỡ một trong hai phiên bản trên và cài bản còn lại thì cần remove như sau:

```bash
Đối với bản GPU thì ta cần remove:
conda remove tensorflow-gpu
Đối với bản CPU thì ta cần remove:
conda remove tensorflow-mkl

Sau đó cài lại như trên nếu không sẽ chạy rất lâu (conda bug)
```

Bước 3:Cài đặt cocoapi

```bash
git clone https://github.com/cocodataset/cocoapi.git
```

Sửa đổi file cocoapi\PythonAPI\\`setup.py`

```python
extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99']
đổi thành
extra_compile_args=[]
```

Chạy vào file cocoapi\PythonAPI

```bash
cd cocoapi\PythonAPI
python setup.py build_ext install
```

Bước 4:Download model đã được training.

[Model](https://github.com/thanh125643/LitterMask/releases/tag/0.1)

Download về giải nén vào module\detector

# Chạy

```bash
python app.py
```

Ứng dụng sẽ chạy trên địa chỉ sau http://127.0.0.1:5000/
