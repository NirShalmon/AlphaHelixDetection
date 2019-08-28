import requests
import lxml.html
import urllib
import os
import re
import pickle

PROJECT_PATH = 'C:\\Bioinformatics\\workshop'  # needs to be changed according to where we save the project


def get_family(family_link, dir_path, proteins):  # link to a SCOPe family
    print(family_link)
    species = set()
    r = requests.get(family_link)
    doc = lxml.html.fromstring(r.content)
    # first, build a set of linjs to species
    for species_link in doc.xpath("//div[@class = 'col-md-12 compact']//td[@class = 'descbox'][contains(./text(),'Species')]"
                                  "//a/@href[contains(.,'https://scop.berkeley.edu/sunid')]"):
        species.add(species_link)
    # now, find relevant proteins
    for species_link in species:
        r = requests.get(species_link)
        doc = lxml.html.fromstring(r.content)
        if not doc.xpath("//text()[contains(.,'not a true') or contains(.,'Hypothetical')]"):
            for prot in doc.xpath("//div[@class = 'col-md-12 compact']//li[contains(./text(),'Domain')]"
                                  "//a[contains(./@href,'https://scop.berkeley.edu/pdb/code=')][following::ul/@class='browse']"):
                if not prot.xpath("following::td[2]//text()[contains(.,'Other') or contains(.,'DNA') or contains(., 'mutant') or contains(., 'automated')]"):
                    prot_id = prot.xpath("@href")[0][-4:]
                    if prot_id not in proteins and not os.path.exists(os.path.join(dir_path, prot_id+'.pdb')) and prot_id != '1zn0' and prot_id != '1ae4':  # if proteins hasn't already been downloaded, download it
                        print(prot_id)  # to be deleted later
                        urllib.request.urlretrieve('http://files.rcsb.org/download/' + prot_id + '.pdb', os.path.join(dir_path, prot_id+'.pdb'))
                        print('downloaded')
                        proteins.add(prot_id)
                        break  # intersted only in up to one PDB entry of each protein


def get_superfamily(superfamily_link, save_path=PROJECT_PATH, create_dir=True):  # a link to a SCOPe superfamily
    r = requests.get(superfamily_link)
    doc = lxml.html.fromstring(r.content)
    superfamily_dir_created = False
    spfamily_name = doc.xpath("//title/text()")[0]
    srch = re.search('Superfamily', spfamily_name)
    if not srch:
        srch = re.search('Family', spfamily_name)
    if not srch:
        return
    spfamily_name = spfamily_name[srch.span()[0]:].replace(":", "-").replace('"', "_").replace("/", "-")

    # create a directory for the superfamily
    if create_dir:
        save_path = os.path.join(save_path,spfamily_name)
        try:
            os.mkdir(save_path)
            superfamily_dir_created = True
        except:
            pass

    # first, build a set of links to all families in superfamily
    families = set()
    for family_link in doc.xpath("//div[@class = 'col-md-12 compact']//ol[@class = 'browse']//tr[not(contains(.//i/text(), 'not a true'))]"
                                 "//a/@href[contains(.,'https://scop.berkeley.edu/sunid')]"):
        families.add(family_link)
    # now, get their proteins
    proteins = set()
    for family_link in families:
        get_family(family_link, save_path, proteins)

    if superfamily_dir_created and not proteins:  # an empty folder
        os.rmdir(save_path)

    return True  # let know that function finished successfully


def get_class(class_link, create_subdirs=True):  # a link to a SCOPe class
    r = requests.get(class_link)
    doc = lxml.html.fromstring(r.content)
    dir_name = doc.xpath("//title/text()")[0]
    dir_name = dir_name[re.search('Class', dir_name).span()[0]:].replace(":","-").replace('"',"'").replace("/","-")
    save_path = os.path.join(PROJECT_PATH,dir_name)  # create a directory for class
    try:
        os.mkdir(save_path)
    except:
        pass

    try:  # get superfamilies links only on first run
        superfamilies = pickle.load(open(dir_name+"_superfamilies.p", "rb"))
    except (OSError, IOError) as e:
        # get all folds
        folds = set()
        for fold_link in doc.xpath(
                "//div[@class = 'col-md-12 compact']//ol[@class = 'browse']//tr[not(contains(.//i/text(), 'not a true'))]"
                "//a/@href[contains(.,'https://scop.berkeley.edu/sunid')]"):
            folds.add(fold_link)
        # get all superfamilies and get proteins from them
        superfamilies = set()
        for fold_link in folds:
            r = requests.get(fold_link)
            doc = lxml.html.fromstring(r.content)
            for superfamily_link in doc.xpath("//div[@class = 'col-md-12 compact']//ol[@class = 'browse']//"
                                              "tr[not(contains(.//i/text(), 'not a true')) and not(contains(.//i/text(),'duplication'))]"
                                              "//a/@href[contains(.,'https://scop.berkeley.edu/sunid')]"):
                superfamilies.add(superfamily_link)
        pickle.dump(superfamilies, open(dir_name+"_superfamilies.p", "wb"))
    # remember which superfamilies were parsed on earlier runs
    try:
        successfully_created = pickle.load(open(dir_name+"_created.p", "rb"))
    except:
        successfully_created = set()
    count = 0  # to be deleted
    for superfamily_link in superfamilies:
        if superfamily_link not in successfully_created:
            if get_superfamily(superfamily_link, save_path, create_subdirs):
                successfully_created.add(superfamily_link)
                pickle.dump(successfully_created, open(dir_name + "_created.p", "wb"))
        count += 1  # to be deleted
        print(count)  # to be deleted


if __name__ == '__main__':
    # get_superfamily('https://scop.berkeley.edu/sunid=46689')

    # all alpha
    get_class('https://scop.berkeley.edu/sunid=46456')

    # alpha and beta
    get_class('https://scop.berkeley.edu/sunid=51349', create_subdirs=False)
