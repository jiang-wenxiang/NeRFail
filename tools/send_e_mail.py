import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header


def send_dict(title, dict_msg: dict):
    # 1. 连接邮箱服务器
    con = smtplib.SMTP_SSL('smtp.qq.com', 465)
    # 2. 登录邮箱
    con.login('your-email-address', 'your-email-smtp-password')
    # 2. 准备数据
    # 创建邮件对象
    msg = MIMEMultipart()
    # 设置邮件主题
    subject = Header(title, 'utf-8').encode()
    msg['Subject'] = subject
    # 设置邮件发送者
    msg['From'] = 'your-email-address <your-email-address>'
    # 设置邮件接受者
    msg['To'] = 'your-email-address'
    # 添加⽂文字内容

    text_str = "<table>"
    for k in dict_msg:
        text_str += "<tr>"
        text_str += "<td>" + str(k) + "</td><td>" + str(dict_msg[k]) + "</td>"
        text_str += "</tr>"
    text_str += "</table>"

    text = MIMEText(text_str, 'html', 'utf-8')
    msg.attach(text)
    # 3.发送邮件
    con.sendmail('your-email-address', 'your-email-address', msg.as_string())
    con.quit()
