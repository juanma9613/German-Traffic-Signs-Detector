import click

import urllib.request
import zipfile
#url='http://www.mynikko.com/dummy/dummy12.zip'
url='http://file-examples.com/wp-content/uploads/2017/02/zip_5MB.zip'

# Download the file from `url` and save it locally under `file_name`:
urllib.request.urlretrieve(url, 'file5mb.zip')
@click.group()
def main():
    pass

@main.command("download")
def download():
    print("here I should be downloading the data...")
    url='http://file-examples.com/wp-content/uploads/2017/02/zip_5MB.zip'
    urllib.request.urlretrieve(url, '/images/file5mb.zip')


    
@main.command("train")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def train(model, directory):
    msg = "I should be training model {} with data from directory {}"
    print(msg.format(model, directory))

@main.command("test")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def test(model, directory):
    msg = "I should be testing model {} with data from directory {}"
    print(msg.format(model, directory))   

@main.command("infer")
@click.option('-m', '--model')
@click.option('-d', '--directory')
def infer(model, directory):
    msg = "I should be infering using model {} with data from directory {}"
    print(msg.format(model, directory))

    
if __name__ == '__main__':
    main(obj={})