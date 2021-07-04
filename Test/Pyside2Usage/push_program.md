### 发布Qt程序_以Pyinstaller为例

`pyinstaller 主程序.py --noconsole --hidden -import PySide2.QtXml`
- 多个文件，只选入口文件
- pyinstaller不会分析资源文件，需要拷贝到对应位置
- 在调试的时候不要添加--noconsole
- PySide2.QtXml是动态导入的(__import__)