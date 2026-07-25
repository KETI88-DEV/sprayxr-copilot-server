/// <reference types="vite/client" />

declare global {
  interface Window {
    __DEMO_API_BASE_URL__?: string
  }

  interface ImportMetaEnv {
    readonly VITE_API_BASE_URL?: string
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv
  }
}

export {}
