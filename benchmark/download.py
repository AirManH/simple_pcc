from urllib.request import urlretrieve
from zipfile import ZipFile


if __name__ == "__main__":
    datasets = {"andrew9", "david9", "phil9", "ricardo9", "sarah9"}
    for dataset in datasets:
        url = "http://plenodb.jpeg.org/pc/microsoft/{}.zip".format(dataset)
        zip_name = "{}.zip".format(dataset)
        zip_dir = "."
        # Download the zip
        urlretrieve(url, filename=zip_name)
        # Unzip
        with ZipFile(zip_name, "r") as zip_f:
            zip_f.extractall(zip_dir)
