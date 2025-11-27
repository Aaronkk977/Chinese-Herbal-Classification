#!/usr/bin/env python3
"""
fix_images.py
- Attempts to open and re-save images listed in corrupt_images.log (or scans dataset)
- On success: backups original to backup_fixed_images/
- On failure: moves file to corrupt_images/
"""
import os
import shutil
import argparse
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def try_fix(path, backup_dir, corrupt_dir):
    try:
        with Image.open(path) as im:
            im.load()
            # Handle palette with transparency
            if im.mode == 'P' and 'transparency' in im.info:
                im = im.convert('RGBA')
            if im.mode == 'RGBA':
                background = Image.new('RGB', im.size, (255,255,255))
                background.paste(im, mask=im.split()[3])
                im = background
            elif im.mode != 'RGB':
                im = im.convert('RGB')

            tmp = path + '.reencode'
            im.save(tmp, format='JPEG', quality=95)
            # Backup original and replace
            os.makedirs(backup_dir, exist_ok=True)
            shutil.move(path, os.path.join(backup_dir, os.path.basename(path)))
            shutil.move(tmp, path)
            return True, None
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-log', type=str, default='corrupt_images.log')
    parser.add_argument('--root', type=str, default='data_split')
    parser.add_argument('--backup-dir', type=str, default='backup_fixed_images')
    parser.add_argument('--corrupt-dir', type=str, default='corrupt_images')
    args = parser.parse_args()

    files = []
    if os.path.exists(args.from_log):
        with open(args.from_log, 'r') as f:
            for line in f:
                if '|' in line:
                    p = line.split('|',1)[0].strip()
                else:
                    p = line.strip()
                if p:
                    files.append(p)
    else:
        for root, _, fnames in os.walk(args.root):
            for fn in fnames:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append(os.path.join(root, fn))

    print(f"Found {len(files)} files to process.")

    os.makedirs(args.backup_dir, exist_ok=True)
    os.makedirs(args.corrupt_dir, exist_ok=True)

    for p in files:
        if not os.path.exists(p):
            print('MISSING:', p)
            continue
        ok, err = try_fix(p, args.backup_dir, args.corrupt_dir)
        if ok:
            print('FIXED:', p)
        else:
            print('BAD:', p, '->', err)
            dest = os.path.join(args.corrupt_dir, os.path.basename(p))
            i = 1
            base, ext = os.path.splitext(dest)
            while os.path.exists(dest):
                dest = f"{base}_{i}{ext}"
                i += 1
            shutil.move(p, dest)
            with open('corrupt_images.log', 'a') as lf:
                lf.write(f"{dest}  # error: {err}\n")

if __name__ == '__main__':
    main()
