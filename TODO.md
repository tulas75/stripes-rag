# TODO

## Cloud storage support
- Index files from SharePoint / OneDrive / Google Drive
- Approach: use `rclone mount` to expose cloud storage as a local filesystem
  - No code changes needed — `stripes index <mount_point>` works as-is
  - `rclone mount onedrive: ~/mnt/onedrive --vfs-cache-mode full`
  - `--vfs-cache-mode full` required so Docling can random-access PDF/XLSX files
- Auth: OAuth2 via `rclone config` (browser flow), or service account for headless servers
- Supported providers: OneDrive, SharePoint, Google Drive, S3, and many more
