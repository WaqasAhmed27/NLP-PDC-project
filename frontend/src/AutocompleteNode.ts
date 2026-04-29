import {
  $applyNodeReplacement,
  TextNode,
  type DOMConversionMap,
  type DOMExportOutput,
  type EditorConfig,
  type LexicalNode,
  type NodeKey,
  type SerializedTextNode,
  type Spread,
} from 'lexical'

export type SerializedAutocompleteNode = Spread<
  {
    type: 'autocomplete'
    version: 1
  },
  SerializedTextNode
>

export class AutocompleteNode extends TextNode {
  static getType(): string {
    return 'autocomplete'
  }

  static clone(node: AutocompleteNode): AutocompleteNode {
    return new AutocompleteNode(node.__text, node.__key)
  }

  static importJSON(serializedNode: SerializedAutocompleteNode): AutocompleteNode {
    return $createAutocompleteNode(serializedNode.text)
  }

  static importDOM(): DOMConversionMap | null {
    return null
  }

  constructor(text: string, key?: NodeKey) {
    super(text, key)
  }

  createDOM(config: EditorConfig): HTMLElement {
    const dom = super.createDOM(config)
    dom.className = 'autocomplete-node'
    dom.contentEditable = 'false'
    dom.style.userSelect = 'none'
    return dom
  }

  updateDOM(prevNode: this, dom: HTMLElement, config: EditorConfig): boolean {
    const shouldReplace = super.updateDOM(prevNode, dom, config)
    dom.contentEditable = 'false'
    dom.style.userSelect = 'none'
    return shouldReplace
  }

  exportDOM(): DOMExportOutput {
    const element = document.createElement('span')
    element.className = 'autocomplete-node'
    element.contentEditable = 'false'
    element.style.userSelect = 'none'
    element.textContent = this.getTextContent()
    return { element }
  }

  exportJSON(): SerializedAutocompleteNode {
    return {
      ...super.exportJSON(),
      type: 'autocomplete',
      version: 1,
    }
  }

  isTextEntity(): true {
    return true
  }

  canInsertTextBefore(): false {
    return false
  }

  canInsertTextAfter(): false {
    return false
  }
}

export function $createAutocompleteNode(text: string): AutocompleteNode {
  return $applyNodeReplacement(new AutocompleteNode(text))
}

export function $isAutocompleteNode(
  node: LexicalNode | null | undefined,
): node is AutocompleteNode {
  return node instanceof AutocompleteNode
}
