from pathlib import Path
import zipfile
root=Path(__file__).resolve().parent
with zipfile.ZipFile(root/'class60_cap_compare_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['cap15','cap30']:
  folder=root/'runs'/name
  for required in ['best.pt','config.json','history.json','baseline.json']:
   p=folder/required
   if not p.is_file():raise FileNotFoundError(p)
  for p in folder.rglob('*'):
   if p.is_file():z.write(p,p.relative_to(root))
 for name in ['train.py','data.py','model_utils.py','external_manifest.json','cap15.log','cap30.log','README.md']:
  z.write(root/name,name)
print('Return: '+str(root/'class60_cap_compare_results.zip'))
