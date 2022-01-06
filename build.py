import os
import shutil
import sys
from distutils.core import Extension, setup

from Cython.Build import cythonize


def walk(root, exclude=(), copy=()):
    exclude.append(os.path.relpath(__file__, root))
    exclude = [os.path.join(root, e) for e in exclude]
    copy = [os.path.join(root, c) for c in copy]

    file_list, dirs_list, copy_list = [], [], []
    if os.path.isdir(root):
        dirs_list.append(root)
    else:
        filename, extension = os.path.splitext(os.path.split(root)[1])
        if extension in ('.py', '.pyx') or filename not in ('__init__',):
            file_list.append(root)
        elif extension not in ('.pyc',):
            copy_list.append(root)
        else:
            pass

    while len(dirs_list) > 0:
        current_dir = dirs_list[0]
        for obj in sorted(os.listdir(current_dir)):
            obj_path = os.path.join(current_dir, obj)
            if obj_path in exclude:
                continue
            if os.path.isdir(obj_path):
                dirs_list.append(obj_path)
            else:
                filename, extension = os.path.splitext(os.path.split(obj_path)[1])
                if extension in ('.py', '.pyx') and filename not in ('__init__',) and obj_path not in copy:
                    file_list.append(obj_path)
                elif extension in ('.pyc',) and obj_path not in copy:
                    pass
                else:
                    copy_list.append(obj_path)
        dirs_list = dirs_list[1:]

    return file_list, copy_list


def build():
    comp_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    flag_single = False
    if os.path.isdir(comp_dir):
        root = os.path.join(sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else comp_dir
    else:
        flag_single = True
        root = comp_dir
        comp_dir = os.path.split(comp_dir)[0]
    comp_dir = os.path.abspath(comp_dir)
    root = os.path.abspath(root)

    build_dirname = 'build'
    temp_dirname = 'temp'
    if not flag_single:
        comp_dirname = os.path.split(comp_dir)[1]
        build_dir = os.path.join(comp_dir, '..', comp_dirname + '_' + build_dirname)
        temp_dir = os.path.join(comp_dir, '..', comp_dirname + '_' + temp_dirname)
    else:
        build_dir = comp_dir
        temp_dir = os.path.join(comp_dir, temp_dirname)

    code_files, copy_files = walk(
        root=root,
        exclude=[build_dirname, temp_dirname, '.git', '.idea', 'save', 'test'],
        copy=('log.py', 'run.py', 'data.py')
    )
    try:
        extension_list = []
        for code_file in code_files:
            path, filename = os.path.split(os.path.relpath(code_file, comp_dir))
            name = path.replace(os.path.sep, '.') + '.' + os.path.splitext(filename)[0]
            if name.startswith('.'):
                name = name[1:]
            e = Extension(name, [code_file], extra_compile_args=["-Os", "-g0"], extra_link_args=["-Wl,--strip-all"])
            extension_list.append(e)
        setup(ext_modules=cythonize(extension_list, language_level=3),
              script_args=["build_ext", "-b", build_dir, "-t", temp_dir])
        print("Copying ...")
        for copy_file in copy_files:
            path, filename = os.path.split(copy_file)
            dst_path = os.path.join(build_dir, os.path.relpath(path, comp_dir))
            os.makedirs(dst_path, exist_ok=True)
            shutil.copyfile(copy_file, os.path.join(dst_path, filename))
    except Exception as ex:
        print("Error: ", ex)
    finally:
        print("Cleaning ...")
        for code_file in code_files:
            path, filename = os.path.split(code_file)
            filename = os.path.splitext(filename)[0]
            c_file = os.path.join(path, filename) + '.c'
            if os.path.exists(c_file):
                os.remove(c_file)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    build()
