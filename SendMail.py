# coding: utf-8
__author__ = 'LiNing'
'''
Message
+- MIMEBase
   +- MIMEMultipart
   +- MIMENonMultipart
      +- MIMEMessage
      +- MIMEText
      +- MIMEImage
'''
import os
from email import encoders
from email.header import Header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.message import MIMEMessage
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import parseaddr, formataddr, formatdate
import smtplib

def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr(
        (Header(name, 'utf-8').encode(),
         addr.encode('utf-8') if isinstance(addr, unicode) else addr)
    )

def send_mail(smtp_server, from_addr, passwd, to_addr, subject, text, files=[]):
    ## --------------------------------------------------------------------------------------
    assert type(to_addr) == list
    assert type(files) == list

    ## --------------------------------------------------------------------------------------
    '''plain or html'''
    msg = MIMEText(text, 'plain', 'utf-8')
    # msg = MIMEText('<html><body>'+'<h1>'+text+'</h1>'+'</body></html>', 'html', 'utf-8')
    ## --------------------------------------------------------------------------------------
    '''plain and html'''
    # msg = MIMEMultipart('alternative')
    # msg.attach(MIMEText(text, 'plain', 'utf-8'))
    # msg.attach(MIMEText('<html><body>'+'<h1>'+text+'</h1>'+'</body></html>', 'html', 'utf-8')) # 附件不嵌入正文
    # # msg.attach(MIMEText('<html><body>'+'<h1>'+text+'</h1>'+'<p><img src="cid:0"></p>'+'</body></html>', 'html', 'utf-8')) # 附件嵌入正文
    # for file in files:
    #     part = MIMEBase('application', 'octet-stream') # 'octet-stream': binary data
    #     part.set_payload(open(file, 'rb').read())
    #     encoders.encode_base64(part) # 用Base64编码
    #     part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file))
    #     part.add_header('Content-ID', '<0>')
    #     part.add_header('X-Attachment-Id', '0')
    #     msg.attach(part) # 添加到MIMEMultipart

    msg['From'] = _format_addr('Me <%s>' % from_addr)
    msg['To'] = _format_addr('You <%s>' % ','.join(to_addr))
    msg['Subject'] = Header(subject, 'utf-8').encode()
    msg['Date'] = formatdate(localtime=True)

    ## --------------------------------------------------------------------------------------
    ## 基于SSL安全连接，Gmail提供的SMTP服务必须要加密传输
    server = smtplib.SMTP(smtp_server, 25) # SMTP协议默认端口是25
    # server.starttls() # 创建安全连接，对于'smtp.gmail.com'，端口587
    server.set_debuglevel(1) # 打印信息
    server.login(from_addr, passwd)
    server.sendmail(from_addr, to_addr, msg.as_string())
    server.quit()

if __name__ == '__main__':
    smtp_server = 'smtp.163.com'
    from_addr = 'xxxx@163.com'
    passwd = 'xxxx'
    to_addr = ['xxxx@163.com']
    subject = 'Hello...'
    text = 'Hello, send by Python...'
    files = []
    send_mail(smtp_server, from_addr, passwd, to_addr, subject, text, files)
