default:
  just --list

clean_cache:
  fd --no-ignore --extension pickle --exec rm {}
