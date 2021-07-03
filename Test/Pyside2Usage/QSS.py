"""由于ui界面使用xml编写，所以可以用CSS的变形QSS进行样式变换
    - 属性StyleSheet

    QPushButton {
        color: red;
        font-size: 15px;
    }
"""

"""Selector
    1. 所有元素 *
    2. 具体类型 QPushButton(包含继承的子类)
    3. .QPushButton(不包含子类)
    4. 使用ID：QPushButton#okButton，找出所有对象名字为okButton的QPushButton
    5. QPushButton[flat='false']选择所有flat为false的QPushButton
    6. QDialog QPushButton: 选择所有QDialog内部QPushButton类型
    7. QDialog > QPushButton: 选择所有QDialog直接子节点QPushButton类型
        - 重孙子就不行了
    
    8. QPushButton:hover {color:red} 当鼠标在元素上方时，显示的样式
    9. QPushButton:disabled {color:red} 指定一个元素为disable时的状态
    10. QCheckBox:hover:checked {color: white} 元素鼠标悬浮，并且处于勾选
"""

"""字体
    {
        font-family: 微软雅黑;
        font-size:15px;
        color: #1d..;
    }
"""

"""背景
    {
        background-color: yellow;
        background-color: rgb(255,255,255)
        
        background-image: url(ggo3.png);
    }
"""

"""边框
    {
        border:1px solid #1d649c;注意空格
            - solid是实线, dashed虚线, dotted点线
        border:none; 无边框
    }
"""

"""宽度、高度
    {
        width: 50px;
        height: 20px;
    }
"""

"""margin, padding

    - margin: 元素与周边元素的边界(外)
    - padding: 元素内容与元素的边界(内)
    {
        margin: 10px 11px 12px 13px
        margin: 10px
        - 上下左右，可指一个
    }
"""

