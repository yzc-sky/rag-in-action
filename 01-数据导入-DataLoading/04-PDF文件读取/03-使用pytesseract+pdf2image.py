# 扫描图片型 PDF，建议用 pytesseract + pdf2image  
# sudo apt-get install tesseract-ocr
# sudo apt-get install tesseract-ocr-chi-sim

import pdf2image
import pytesseract

# 将 PDF 转换为图片并保存
images = pdf2image.convert_from_path('90-文档-Data/黑悟空/黑神话悟空.pdf')
for i, image in enumerate(images):
    image.save(f'output/page_{i+1}.png')

# 使用 pytesseract 提取文本
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang='chi_sim')
    print(f"第 {i+1} 页文本:")
    print(text)
    print("\n") 