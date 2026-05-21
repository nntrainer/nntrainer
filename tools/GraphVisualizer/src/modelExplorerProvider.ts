import * as vscode from 'vscode';
import { ConversionResult } from './types';

export class ModelExplorerProvider implements vscode.TreeDataProvider<ExplorerItem> {
    private result: ConversionResult | null = null;
    private _onDidChangeTreeData = new vscode.EventEmitter<ExplorerItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    setConversionResult(result: ConversionResult) {
        this.result = result;
        this._onDidChangeTreeData.fire(undefined);
    }

    getConversionResult(): ConversionResult | null {
        return this.result;
    }

    getTreeItem(element: ExplorerItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ExplorerItem): ExplorerItem[] {
        if (!this.result) {
            return [new ExplorerItem('No model loaded', '', vscode.TreeItemCollapsibleState.None)];
        }

        if (!element) {
            // Root items
            const items: ExplorerItem[] = [];

            // Model info
            if (this.result.modelStructure) {
                const s = this.result.modelStructure;
                items.push(new ExplorerItem(
                    `Model: ${s.model_type} (${s.arch_type})`,
                    'model-info',
                    vscode.TreeItemCollapsibleState.Collapsed
                ));
            }

            // Layers summary
            items.push(new ExplorerItem(
                `NNTrainer Layers (${this.result.nntrainerLayers.length})`,
                'layers',
                vscode.TreeItemCollapsibleState.Collapsed
            ));

            // FX Graph summary
            items.push(new ExplorerItem(
                `FX Graph Nodes (${this.result.fxGraph.length})`,
                'fx-nodes',
                vscode.TreeItemCollapsibleState.Collapsed
            ));

            // Verification
            const issues = this.result.unsupportedOps.length + this.result.unknownLayers.length;
            items.push(new ExplorerItem(
                `Verification${issues > 0 ? ` (${issues} issues)` : ' (OK)'}`,
                'verification',
                vscode.TreeItemCollapsibleState.Collapsed
            ));

            return items;
        }

        // Children based on context
        switch (element.contextValue) {
            case 'model-info':
                return this.getModelInfoChildren();
            case 'layers':
                return this.getLayerChildren();
            case 'fx-nodes':
                return this.getFxNodeChildren();
            case 'verification':
                return this.getVerificationChildren();
            default:
                return [];
        }
    }

    private getModelInfoChildren(): ExplorerItem[] {
        const s = this.result!.modelStructure!;
        return [
            new ExplorerItem(`vocab_size: ${s.vocab_size}`, ''),
            new ExplorerItem(`hidden_size: ${s.hidden_size}`, ''),
            new ExplorerItem(`num_layers: ${s.num_layers}`, ''),
            new ExplorerItem(`num_heads: ${s.num_heads}`, ''),
            new ExplorerItem(`num_kv_heads: ${s.num_kv_heads}`, ''),
            new ExplorerItem(`head_dim: ${s.head_dim}`, ''),
            new ExplorerItem(`intermediate_size: ${s.intermediate_size}`, ''),
        ];
    }

    private getLayerChildren(): ExplorerItem[] {
        return this.result!.nntrainerLayers.map(l =>
            new ExplorerItem(
                `${l.name} [${l.layer_type}]`,
                'layer-item',
                vscode.TreeItemCollapsibleState.None
            )
        );
    }

    private getFxNodeChildren(): ExplorerItem[] {
        return this.result!.fxGraph
            .filter(n => n.op !== 'placeholder' && n.op !== 'output')
            .map(n =>
                new ExplorerItem(
                    `${n.name} [${n.op}: ${n.target}]`,
                    'fx-item',
                    vscode.TreeItemCollapsibleState.None
                )
            );
    }

    private getVerificationChildren(): ExplorerItem[] {
        const items: ExplorerItem[] = [];
        const mapped = this.result!.nodeMapping.filter(m => m.mappingType === 'direct').length;
        const total = this.result!.fxGraph.filter(n => n.op !== 'placeholder' && n.op !== 'output').length;
        items.push(new ExplorerItem(`Mapped: ${mapped}/${total}`, ''));

        for (const op of this.result!.unsupportedOps) {
            items.push(new ExplorerItem(`Unsupported: ${op}`, ''));
        }
        for (const l of this.result!.unknownLayers) {
            items.push(new ExplorerItem(`Unknown: ${l}`, ''));
        }
        return items;
    }
}

class ExplorerItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly contextValue: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState = vscode.TreeItemCollapsibleState.None
    ) {
        super(label, collapsibleState);
        this.contextValue = contextValue;
    }
}
