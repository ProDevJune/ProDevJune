#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ 제목을 입력하세요. 예: ./new-post.sh hello-world"
  exit 1
fi

TITLE_SLUG=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
DATE=$(date '+%Y-%m-%d')
TIME=$(date '+%Y-%m-%d %H:%M:%S %z')
FILENAME="_posts/$DATE-$TITLE_SLUG.md"

# 템플릿에서 복사
cp post-template.md "$FILENAME"

# 날짜 자동 삽입
sed -i "s/date: .*/date: $TIME/" "$FILENAME"

echo "✅ 새 포스트 생성됨: $FILENAME"
