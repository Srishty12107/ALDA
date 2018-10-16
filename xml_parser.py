import xml.etree.ElementTree as ET
tree = ET.parse('CFR-2016-title5-vol3.xml')
root = tree.getroot()
f=open('title_vol.txt','a+')
#result = ''
for element in root.iter():
        try:    
            if type(element.text) == str:    
                   f.write(element.text)
        except TypeError:    
               element.text=""
          
#result = result[:-1] # Remove trailing space

#f=open('title1_vol1.txt','r')


