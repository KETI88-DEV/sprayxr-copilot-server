export {}

declare global {
  interface Window {
    readonly desktop: Readonly<{ platform: NodeJS.Platform; electronVersion: string }>
  }
}
