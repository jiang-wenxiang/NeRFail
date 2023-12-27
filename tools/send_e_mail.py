import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header


def send_dict(title, dict_msg: dict):
    smtp_server_address = 'smtp.qq.com'
    smtp_server_port = 465
    your_email_address = "change-for-yourself-email"
    your_email_password = "change-for-yourself-email-smtp-password"

    if (your_email_address != "change-for-yourself-email" and
            your_email_password != "change-for-yourself-email-smtp-password"):

        # 1. 连接邮箱服务器
        con = smtplib.SMTP_SSL(smtp_server_address, smtp_server_port)
        # 2. 登录邮箱
        con.login(your_email_address, your_email_password)
        # 2. 准备数据
        # 创建邮件对象
        msg = MIMEMultipart()
        # 设置邮件主题
        subject = Header(title, 'utf-8').encode()
        msg['Subject'] = subject
        # 设置邮件发送者
        msg['From'] = your_email_address+' <'+your_email_address+'>'
        # 设置邮件接受者
        msg['To'] = your_email_address
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
        con.sendmail(your_email_address, your_email_address, msg.as_string())
        con.quit()
