use crate::model::{Page, ProcessedDocument};

pub(crate) fn to_xml(doc: &ProcessedDocument) -> String {
    let mut out = String::new();
    out.push_str("<doc v=\"1\" src=\"");
    escape_into(&doc.source, &mut out);
    out.push_str("\" pc=\"");
    out.push_str(&doc.page_count.to_string());
    out.push_str("\">");

    out.push_str("<meta");
    if let Some(title) = &doc.metadata.title {
        out.push_str(" title=\"");
        escape_into(title, &mut out);
        out.push('\"');
    }
    if let Some(author) = &doc.metadata.author {
        out.push_str(" author=\"");
        escape_into(author, &mut out);
        out.push('\"');
    }
    if let Some(date) = &doc.metadata.creation_date {
        out.push_str(" cdate=\"");
        escape_into(date, &mut out);
        out.push('\"');
    }
    out.push_str("/>");

    out.push_str("<pages>");
    for page in &doc.pages {
        write_page(page, &mut out);
    }
    out.push_str("</pages>");

    out.push_str("<chunks>");
    for chunk in &doc.chunks {
        out.push_str("<c id=\"");
        escape_into(&chunk.id, &mut out);
        out.push_str("\" ps=\"");
        out.push_str(&chunk.page_start.to_string());
        out.push_str("\" pe=\"");
        out.push_str(&chunk.page_end.to_string());
        out.push_str("\" te=\"");
        out.push_str(&chunk.token_estimate.to_string());
        out.push_str("\" md=\"");
        out.push_str(chunk.modality.as_str());
        out.push_str("\">");
        escape_into(&chunk.content, &mut out);
        out.push_str("</c>");
    }
    out.push_str("</chunks>");

    out.push_str("</doc>");
    out
}

fn write_page(page: &Page, out: &mut String) {
    out.push_str("<p i=\"");
    out.push_str(&page.index.to_string());
    out.push_str("\" sp=\"");
    out.push_str(&page.source_page.to_string());
    out.push_str("\" m=\"");
    out.push_str(page.mode.as_str());
    out.push_str("\" tc=\"");
    out.push_str(&page.text_chars.to_string());
    out.push_str("\" te=\"");
    out.push_str(&page.token_estimate.to_string());
    out.push_str("\">");

    if !page.text.is_empty() {
        out.push_str("<t>");
        escape_into(&page.text, out);
        out.push_str("</t>");
    }
    if let Some(img) = &page.image_ref {
        out.push_str("<img src=\"");
        escape_into(img, out);
        out.push_str("\"/>");
    }
    if let Some(v) = &page.vision_ref {
        out.push_str("<v r=\"");
        escape_into(v, out);
        out.push_str("\"/>");
    }
    out.push_str("</p>");
}

fn escape_into(input: &str, out: &mut String) {
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{Chunk, ChunkModality, DocumentMeta, Page, PageMode, ProcessedDocument};

    use super::to_xml;

    #[test]
    fn xml_is_deterministic() {
        let doc = ProcessedDocument {
            source: "sample.pdf".to_string(),
            page_count: 1,
            metadata: DocumentMeta {
                title: Some("T".to_string()),
                author: Some("A".to_string()),
                creation_date: None,
            },
            pages: vec![Page {
                index: 1,
                source_page: 1,
                mode: PageMode::Native,
                text: "hello".to_string(),
                text_chars: 5,
                token_estimate: 2,
                vision_ref: None,
                image_ref: None,
            }],
            chunks: vec![Chunk {
                id: "p1c0".to_string(),
                page_start: 1,
                page_end: 1,
                token_estimate: 2,
                modality: ChunkModality::Text,
                content: "hello".to_string(),
            }],
        };

        let a = to_xml(&doc);
        let b = to_xml(&doc);
        assert_eq!(a, b);
        assert!(a.contains("<chunks>"));
    }
}
