import PyInstaller.__main__

PyInstaller.__main__.run([
    'src/mini_pixel/__init__.py',
    '-nmini_pixel',
    '--onefile',
    '--add-data=src/mini_pixel/a_Shadow2_MiniPixelDisplay.png:.',
])
