import { contextBridge } from 'electron'

const desktop = Object.freeze({
  platform: process.platform,
  electronVersion: process.versions.electron ?? 'unknown',
})

contextBridge.exposeInMainWorld('desktop', desktop)
