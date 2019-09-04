# -*- mode: python -*-

block_cipher = None


a = Analysis(['storm_selection_gui_ATT5.py'],
             pathex=['/Users/dylan/Box/chl/RSST-master'],
             binaries=[],
             datas=[('NACCS_TS_Sim0_Post0_ST_Stat_SRR.h5', '.'), ('S2G_TS_Sim0_Post0_ST_Stat_SRR_dylan.h5', '.')],
             hiddenimports=['h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy', 'sklearn', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree', 'sklearn.tree._utils'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='Representative Storm Selection Tool (py35_32Bit)',
          debug=False,
          strip=False,
          upx=True,
          console=True )