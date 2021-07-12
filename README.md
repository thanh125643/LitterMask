# LitterMask

Use and research from TACO github.
# Giới thiệu
Dự án nhằm mục đích sử dụng Mask R-CNN cho việc nhận diện phân vùng rác dựa trên tập dữ liệu TACO có sẵn.

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
Buớc 2:Khởi tạo môi trường FlaskTaco từ file env.yml
```bash
conda env create -f env.yml -n FlaskTaco

conda activate FlaskTaco
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

## Lưu ý
Nếu máy không có gpu thì install Tensorflow-mkl
```bash
conda uninstall tensorflow
conda install tensorflow-mkl
```
# Chạy
```bash
python app.py 
```
Ứng dụng sẽ chạy trên địa chỉ sau http://127.0.0.1:5000/