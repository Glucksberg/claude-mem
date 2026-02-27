/**
 * ChromaSync Service
 *
 * Automatically syncs observations and session summaries to ChromaDB via MCP.
 * This service provides real-time semantic search capabilities by maintaining
 * a vector database synchronized with SQLite.
 *
 * Uses ChromaMcpManager to communicate with chroma-mcp over stdio MCP protocol.
 * The chroma-mcp server handles its own embedding and persistent storage,
 * eliminating the need for chromadb npm package and ONNX/WASM dependencies.
 *
 * Design: Fail-fast with no fallbacks - if Chroma is unavailable, syncing fails.
 */

import { ChromaMcpManager } from './ChromaMcpManager.js';
import { ParsedObservation, ParsedSummary } from '../../sdk/parser.js';
import { SessionStore } from '../sqlite/SessionStore.js';
import { Thought } from '../sqlite/thoughts/types.js';
import { logger } from '../../utils/logger.js';

interface ChromaDocument {
  id: string;
  document: string;
  metadata: Record<string, string | number>;
}

interface StoredObservation {
  id: number;
  memory_session_id: string;
  project: string;
  text: string | null;
  type: string;
  title: string | null;
  subtitle: string | null;
  facts: string | null; // JSON
  narrative: string | null;
  concepts: string | null; // JSON
  files_read: string | null; // JSON
  files_modified: string | null; // JSON
  prompt_number: number;
  discovery_tokens: number; // ROI metrics
  created_at: string;
  created_at_epoch: number;
}

interface StoredSummary {
  id: number;
  memory_session_id: string;
  project: string;
  request: string | null;
  investigated: string | null;
  learned: string | null;
  completed: string | null;
  next_steps: string | null;
  notes: string | null;
  prompt_number: number;
  discovery_tokens: number; // ROI metrics
  created_at: string;
  created_at_epoch: number;
}

interface StoredUserPrompt {
  id: number;
  content_session_id: string;
  prompt_number: number;
  prompt_text: string;
  created_at: string;
  created_at_epoch: number;
  memory_session_id: string;
  project: string;
}

export class ChromaSync {
  private project: string;
  private collectionName: string;
  private collectionCreated = false;
  private readonly BATCH_SIZE = 100;

  // Windows popup concern resolved: the worker daemon starts with -WindowStyle Hidden,
  // so child processes (uvx/chroma-mcp) inherit the hidden console and don't create new windows.
  // MCP SDK's StdioClientTransport uses shell:false and no detached flag, so console is inherited.
  private readonly disabled: boolean = false;

  // Connection mutex — coalesces concurrent ensureConnection() calls onto single spawn
  private connectionPromise: Promise<void> | null = null;

  // Circuit breaker — stops retry storms after repeated failures
  private consecutiveFailures: number = 0;
  private circuitOpenUntil: number = 0;
  private static readonly MAX_FAILURES = 3;
  private static readonly CIRCUIT_OPEN_MS = 60_000;

  constructor(project: string) {
    this.project = project;
    this.collectionName = `cm__${project}`;
    this.VECTOR_DB_DIR = path.join(os.homedir(), '.claude-mem', 'vector-db');

    // Disable on Windows to prevent console popups from MCP subprocess spawning
    // The MCP SDK's StdioClientTransport spawns Python processes that create visible windows
    const allowWindowsChroma = process.env.CLAUDE_MEM_ENABLE_CHROMA_WINDOWS === 'true';
    this.disabled = process.platform === 'win32' && !allowWindowsChroma;
    if (this.disabled) {
      logger.warn('CHROMA_SYNC', 'Vector search disabled on Windows (prevents console popups)', {
        project: this.project,
        reason: 'MCP SDK subprocess spawning causes visible console windows'
      });
    } else if (process.platform === 'win32' && allowWindowsChroma) {
      logger.warn('CHROMA_SYNC', 'Vector search enabled on Windows by override', {
        project: this.project,
        note: 'Console popups may occur when spawning MCP subprocesses'
      });
    }
  }

  /**
   * Ensure collection exists in Chroma via MCP.
   * chroma_create_collection is idempotent - safe to call multiple times.
   * Uses collectionCreated flag to avoid redundant calls within a session.
   */
  private getCombinedCertPath(): string | undefined {
    const combinedCertPath = path.join(os.homedir(), '.claude-mem', 'combined_certs.pem');

    // If combined certs already exist and are recent (less than 24 hours old), use them
    if (fs.existsSync(combinedCertPath)) {
      const stats = fs.statSync(combinedCertPath);
      const ageMs = Date.now() - stats.mtimeMs;
      if (ageMs < 24 * 60 * 60 * 1000) {
        return combinedCertPath;
      }
    }

    // Only create on macOS (Zscaler certificate extraction uses macOS security command)
    if (process.platform !== 'darwin') {
      return undefined;
    }

    try {
      // Use uvx to resolve the correct certifi path for the exact Python environment it uses
      // This is more reliable than scanning the uv cache directory structure
      let certifiPath: string | undefined;
      try {
        certifiPath = execSync(
          'uvx --with certifi python -c "import certifi; print(certifi.where())"',
          { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 }
        ).trim();
      } catch {
        // uvx or certifi not available
        return undefined;
      }

      if (!certifiPath || !fs.existsSync(certifiPath)) {
        return undefined;
      }

      // Try to extract Zscaler certificate from macOS keychain
      let zscalerCert = '';
      try {
        zscalerCert = execSync(
          'security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain',
          { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 5000 }
        );
      } catch {
        // Zscaler not found, which is fine - not all environments have it
        return undefined;
      }

      // Validate PEM certificate format (must have both BEGIN and END markers)
      if (!zscalerCert ||
          !zscalerCert.includes('-----BEGIN CERTIFICATE-----') ||
          !zscalerCert.includes('-----END CERTIFICATE-----')) {
        return undefined;
      }

      // Create combined certificate bundle with atomic write (write to temp, then rename)
      const certifiContent = fs.readFileSync(certifiPath, 'utf8');
      const tempPath = combinedCertPath + '.tmp';
      fs.writeFileSync(tempPath, certifiContent + '\n' + zscalerCert);
      fs.renameSync(tempPath, combinedCertPath);
      logger.info('CHROMA_SYNC', 'Created combined SSL certificate bundle for Zscaler', {
        path: combinedCertPath
      });

      return combinedCertPath;
    } catch (error) {
      logger.debug('CHROMA_SYNC', 'Could not create combined cert bundle', {}, error as Error);
      return undefined;
    }
  }

  /**
   * Check if Chroma is disabled (Windows)
   */
  isDisabled(): boolean {
    return this.disabled;
  }

  /**
   * Ensure MCP client is connected to Chroma server
   * Throws error if connection fails
   * No-op on Windows when disabled (defense-in-depth)
   */
  private async ensureConnection(): Promise<void> {
    if (this.disabled) {
      throw new Error('Chroma is disabled on this platform');
    }

    if (this.connected && this.client) {
      return;
    }

    // Circuit breaker: stop retrying after repeated failures
    if (Date.now() < this.circuitOpenUntil) {
      throw new Error('Chroma circuit breaker open — connection disabled for 60s after repeated failures');
    }

    // Connection mutex: coalesce concurrent callers onto single spawn
    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    // Capture reference to detect if a newer call replaced us.
    // Without this, a concurrent caller's promise could be cleared by
    // an older caller's finally{} block — a subtle race condition.
    const p = this.connectionPromise = this._doConnect();
    try {
      await p;
      this.consecutiveFailures = 0; // Reset on success
    } catch (error) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= ChromaSync.MAX_FAILURES) {
        this.circuitOpenUntil = Date.now() + ChromaSync.CIRCUIT_OPEN_MS;
        logger.warn('CHROMA_SYNC', 'Circuit breaker tripped — disabling Chroma for 60s', {
          failures: this.consecutiveFailures
        });
      }
      throw error;
    } finally {
      // Only clear if still the same promise (newer call may have replaced it)
      if (this.connectionPromise === p) {
        this.connectionPromise = null;
      }
    }
  }

  /**
   * Internal connection logic — called only by ensureConnection() mutex
   */
  private async _doConnect(): Promise<void> {
    logger.info('CHROMA_SYNC', 'Connecting to Chroma MCP server...', { project: this.project });

    try {
      const settings = SettingsDefaultsManager.loadFromFile(USER_SETTINGS_PATH);
      const pythonVersion = settings.CLAUDE_MEM_PYTHON_VERSION;
      const combinedCertPath = this.getCombinedCertPath();

      const transportOptions: any = {
        command: 'uvx',
        args: [
          '--python', pythonVersion,
          'chroma-mcp',
          '--client-type', 'persistent',
          '--data-dir', this.VECTOR_DB_DIR
        ],
        stderr: 'ignore'
      };

      if (combinedCertPath) {
        transportOptions.env = {
          ...process.env,
          SSL_CERT_FILE: combinedCertPath,
          REQUESTS_CA_BUNDLE: combinedCertPath,
          CURL_CA_BUNDLE: combinedCertPath
        };
        logger.info('CHROMA_SYNC', 'Using combined SSL certificates for Zscaler compatibility', {
          certPath: combinedCertPath
        });
      }

      this.transport = new StdioClientTransport(transportOptions);

      this.client = new Client({
        name: 'claude-mem-chroma-sync',
        version: packageVersion
      }, {
        capabilities: {}
      });
    } catch (error) {
      // Safe cleanup: close transport before nulling reference
      if (this.transport) {
        try { await this.transport.close(); } catch {}
      }
      this.transport = null;
      this.client = null;
      this.connected = false;
      logger.error('CHROMA_SYNC', 'Failed to connect to Chroma MCP server', { project: this.project }, error as Error);
      throw new Error(`Chroma connection failed: ${error instanceof Error ? error.message : String(error)}`);
    }

    this.collectionCreated = true;

    logger.debug('CHROMA_SYNC', 'Collection ready', {
      collection: this.collectionName
    });
  }

  /**
   * Safely parse a JSON string, returning fallback on failure.
   * Prevents a single corrupted record from crashing the entire backfill.
   */
  private safeJsonParse(value: string | null, fallback: any[] = [], context?: { field: string; obsId: number }): any {
    if (!value) return fallback;
    try {
      return JSON.parse(value);
    } catch (error) {
      // Check if this is a connection error - don't try to create collection
      const errorMessage = error instanceof Error ? error.message : String(error);
      const isConnectionError =
        errorMessage.includes('Not connected') ||
        errorMessage.includes('Connection closed') ||
        errorMessage.includes('MCP error -32000');

      if (isConnectionError) {
        await this.safeResetConnection();
        logger.error('CHROMA_SYNC', 'Connection lost during collection check',
          { collection: this.collectionName }, error as Error);
        throw new Error(`Chroma connection lost: ${errorMessage}`);
      }

      // Only attempt creation if it's genuinely a "collection not found" error
      logger.error('CHROMA_SYNC', 'Collection check failed, attempting to create', { collection: this.collectionName }, error as Error);
      logger.info('CHROMA_SYNC', 'Creating collection', { collection: this.collectionName });

      try {
        await this.client.callTool({
          name: 'chroma_create_collection',
          arguments: {
            collection_name: this.collectionName,
            embedding_function_name: 'default'
          }
        });

        logger.info('CHROMA_SYNC', 'Collection created', { collection: this.collectionName });
      } catch (createError) {
        logger.error('CHROMA_SYNC', 'Failed to create collection', { collection: this.collectionName }, createError as Error);
        throw new Error(`Collection creation failed: ${createError instanceof Error ? createError.message : String(createError)}`);
      }
    }
  }

  /**
   * Format observation into Chroma documents (granular approach)
   * Each semantic field becomes a separate vector document
   */
  private formatObservationDocs(obs: StoredObservation): ChromaDocument[] {
    const documents: ChromaDocument[] = [];

    // Parse JSON fields — tolerant of corrupted data
    const facts = this.safeJsonParse(obs.facts, [], { field: 'facts', obsId: obs.id });
    const concepts = this.safeJsonParse(obs.concepts, [], { field: 'concepts', obsId: obs.id });
    const files_read = this.safeJsonParse(obs.files_read, [], { field: 'files_read', obsId: obs.id });
    const files_modified = this.safeJsonParse(obs.files_modified, [], { field: 'files_modified', obsId: obs.id });

    const baseMetadata: Record<string, string | number> = {
      sqlite_id: obs.id,
      doc_type: 'observation',
      memory_session_id: obs.memory_session_id,
      project: obs.project,
      created_at_epoch: obs.created_at_epoch,
      type: obs.type || 'discovery',
      title: obs.title || 'Untitled'
    };

    // Add optional metadata fields
    if (obs.subtitle) {
      baseMetadata.subtitle = obs.subtitle;
    }
    if (concepts.length > 0) {
      baseMetadata.concepts = concepts.join(',');
    }
    if (files_read.length > 0) {
      baseMetadata.files_read = files_read.join(',');
    }
    if (files_modified.length > 0) {
      baseMetadata.files_modified = files_modified.join(',');
    }

    // Narrative as separate document
    if (obs.narrative) {
      documents.push({
        id: `obs_${obs.id}_narrative`,
        document: obs.narrative,
        metadata: { ...baseMetadata, field_type: 'narrative' }
      });
    }

    // Text as separate document (legacy field)
    if (obs.text) {
      documents.push({
        id: `obs_${obs.id}_text`,
        document: obs.text,
        metadata: { ...baseMetadata, field_type: 'text' }
      });
    }

    // Each fact as separate document
    facts.forEach((fact: string, index: number) => {
      documents.push({
        id: `obs_${obs.id}_fact_${index}`,
        document: fact,
        metadata: { ...baseMetadata, field_type: 'fact', fact_index: index }
      });
    });

    return documents;
  }

  /**
   * Format summary into Chroma documents (granular approach)
   * Each summary field becomes a separate vector document
   */
  private formatSummaryDocs(summary: StoredSummary): ChromaDocument[] {
    const documents: ChromaDocument[] = [];

    const baseMetadata: Record<string, string | number> = {
      sqlite_id: summary.id,
      doc_type: 'session_summary',
      memory_session_id: summary.memory_session_id,
      project: summary.project,
      created_at_epoch: summary.created_at_epoch,
      prompt_number: summary.prompt_number || 0
    };

    // Each field becomes a separate document
    if (summary.request) {
      documents.push({
        id: `summary_${summary.id}_request`,
        document: summary.request,
        metadata: { ...baseMetadata, field_type: 'request' }
      });
    }

    if (summary.investigated) {
      documents.push({
        id: `summary_${summary.id}_investigated`,
        document: summary.investigated,
        metadata: { ...baseMetadata, field_type: 'investigated' }
      });
    }

    if (summary.learned) {
      documents.push({
        id: `summary_${summary.id}_learned`,
        document: summary.learned,
        metadata: { ...baseMetadata, field_type: 'learned' }
      });
    }

    if (summary.completed) {
      documents.push({
        id: `summary_${summary.id}_completed`,
        document: summary.completed,
        metadata: { ...baseMetadata, field_type: 'completed' }
      });
    }

    if (summary.next_steps) {
      documents.push({
        id: `summary_${summary.id}_next_steps`,
        document: summary.next_steps,
        metadata: { ...baseMetadata, field_type: 'next_steps' }
      });
    }

    if (summary.notes) {
      documents.push({
        id: `summary_${summary.id}_notes`,
        document: summary.notes,
        metadata: { ...baseMetadata, field_type: 'notes' }
      });
    }

    return documents;
  }

  /**
   * Add documents to Chroma in batch via MCP
   * Throws error if batch add fails
   */
  private async addDocuments(documents: ChromaDocument[]): Promise<void> {
    if (documents.length === 0) {
      return;
    }

    await this.ensureCollectionExists();

    const chromaMcp = ChromaMcpManager.getInstance();

    // Add in batches
    for (let i = 0; i < documents.length; i += this.BATCH_SIZE) {
      const batch = documents.slice(i, i + this.BATCH_SIZE);

      // Sanitize metadata: filter out null, undefined, and empty string values
      // that chroma-mcp may reject (e.g., null subtitle from raw SQLite rows)
      const cleanMetadatas = batch.map(d =>
        Object.fromEntries(
          Object.entries(d.metadata).filter(([_, v]) => v !== null && v !== undefined && v !== '')
        )
      );

      try {
        await chromaMcp.callTool('chroma_add_documents', {
          collection_name: this.collectionName,
          ids: batch.map(d => d.id),
          documents: batch.map(d => d.document),
          metadatas: cleanMetadatas
        });
      } catch (error) {
        logger.error('CHROMA_SYNC', 'Batch add failed, continuing with remaining batches', {
          collection: this.collectionName,
          batchStart: i,
          batchSize: batch.length
        }, error as Error);
      }
    }

    logger.debug('CHROMA_SYNC', 'Documents added', {
      collection: this.collectionName,
      count: documents.length
    });
  }

  /**
   * Sync a single observation to Chroma
   * Blocks until sync completes, throws on error
   */
  async syncObservation(
    observationId: number,
    memorySessionId: string,
    project: string,
    obs: ParsedObservation,
    promptNumber: number,
    createdAtEpoch: number,
    discoveryTokens: number = 0
  ): Promise<void> {
    // Convert ParsedObservation to StoredObservation format
    const stored: StoredObservation = {
      id: observationId,
      memory_session_id: memorySessionId,
      project: project,
      text: null, // Legacy field, not used
      type: obs.type,
      title: obs.title,
      subtitle: obs.subtitle,
      facts: JSON.stringify(obs.facts),
      narrative: obs.narrative,
      concepts: JSON.stringify(obs.concepts),
      files_read: JSON.stringify(obs.files_read),
      files_modified: JSON.stringify(obs.files_modified),
      prompt_number: promptNumber,
      discovery_tokens: discoveryTokens,
      created_at: new Date(createdAtEpoch * 1000).toISOString(),
      created_at_epoch: createdAtEpoch
    };

    const documents = this.formatObservationDocs(stored);

    logger.info('CHROMA_SYNC', 'Syncing observation', {
      observationId,
      documentCount: documents.length,
      project
    });

    await this.addDocuments(documents);
  }

  /**
   * Sync a single summary to Chroma
   * Blocks until sync completes, throws on error
   */
  async syncSummary(
    summaryId: number,
    memorySessionId: string,
    project: string,
    summary: ParsedSummary,
    promptNumber: number,
    createdAtEpoch: number,
    discoveryTokens: number = 0
  ): Promise<void> {
    // Convert ParsedSummary to StoredSummary format
    const stored: StoredSummary = {
      id: summaryId,
      memory_session_id: memorySessionId,
      project: project,
      request: summary.request,
      investigated: summary.investigated,
      learned: summary.learned,
      completed: summary.completed,
      next_steps: summary.next_steps,
      notes: summary.notes,
      prompt_number: promptNumber,
      discovery_tokens: discoveryTokens,
      created_at: new Date(createdAtEpoch * 1000).toISOString(),
      created_at_epoch: createdAtEpoch
    };

    const documents = this.formatSummaryDocs(stored);

    logger.info('CHROMA_SYNC', 'Syncing summary', {
      summaryId,
      documentCount: documents.length,
      project
    });

    await this.addDocuments(documents);
  }

  /**
   * Format user prompt into Chroma document
   * Each prompt becomes a single document (unlike observations/summaries which split by field)
   */
  private formatUserPromptDoc(prompt: StoredUserPrompt): ChromaDocument {
    return {
      id: `prompt_${prompt.id}`,
      document: prompt.prompt_text,
      metadata: {
        sqlite_id: prompt.id,
        doc_type: 'user_prompt',
        memory_session_id: prompt.memory_session_id,
        project: prompt.project,
        created_at_epoch: prompt.created_at_epoch,
        prompt_number: prompt.prompt_number
      }
    };
  }

  /**
   * Sync a single user prompt to Chroma
   * Blocks until sync completes, throws on error
   */
  async syncUserPrompt(
    promptId: number,
    memorySessionId: string,
    project: string,
    promptText: string,
    promptNumber: number,
    createdAtEpoch: number
  ): Promise<void> {
    // Create StoredUserPrompt format
    const stored: StoredUserPrompt = {
      id: promptId,
      content_session_id: '', // Not needed for Chroma sync
      prompt_number: promptNumber,
      prompt_text: promptText,
      created_at: new Date(createdAtEpoch * 1000).toISOString(),
      created_at_epoch: createdAtEpoch,
      memory_session_id: memorySessionId,
      project: project
    };

    const document = this.formatUserPromptDoc(stored);

    logger.info('CHROMA_SYNC', 'Syncing user prompt', {
      promptId,
      project
    });

    await this.addDocuments([document]);
  }

  /**
   * Sync a single thought to Chroma
   * Creates one document per thought with the thinking text as content
   * No-op on Windows (Chroma disabled to prevent console popups)
   */
  async syncThought(thought: Thought): Promise<void> {
    if (this.disabled) return;

    const document: ChromaDocument = {
      id: `thought_${thought.id}`,
      document: thought.thinking_text,
      metadata: {
        doc_type: 'thought',
        thought_id: thought.id,
        memory_session_id: thought.memory_session_id,
        project: thought.project,
        message_index: thought.message_index ?? 0,
        created_at_epoch: thought.created_at_epoch
      }
    };

    logger.info('CHROMA_SYNC', 'Syncing thought', {
      thoughtId: thought.id,
      project: thought.project
    });

    await this.addDocuments([document]);
  }

  /**
   * Sync multiple thoughts to Chroma in a single batch
   * No-op on Windows (Chroma disabled to prevent console popups)
   */
  async syncThoughts(thoughts: Thought[]): Promise<void> {
    if (this.disabled) return;
    if (thoughts.length === 0) return;

    const documents: ChromaDocument[] = thoughts.map(thought => ({
      id: `thought_${thought.id}`,
      document: thought.thinking_text,
      metadata: {
        doc_type: 'thought',
        thought_id: thought.id,
        memory_session_id: thought.memory_session_id,
        project: thought.project,
        message_index: thought.message_index ?? 0,
        created_at_epoch: thought.created_at_epoch
      }
    }));

    logger.info('CHROMA_SYNC', 'Syncing thoughts batch', {
      count: thoughts.length,
      project: thoughts[0].project
    });

    await this.addDocuments(documents);
  }

  /**
   * Fetch all existing document IDs from Chroma collection
   * Returns Sets of SQLite IDs for observations, summaries, prompts, and thoughts
   */
  private async getExistingChromaIds(projectOverride?: string): Promise<{
    observations: Set<number>;
    summaries: Set<number>;
    prompts: Set<number>;
    thoughts: Set<number>;
  }> {
    const targetProject = projectOverride ?? this.project;
    await this.ensureCollectionExists();

    const chromaMcp = ChromaMcpManager.getInstance();

    const observationIds = new Set<number>();
    const summaryIds = new Set<number>();
    const promptIds = new Set<number>();
    const thoughtIds = new Set<number>();

    let offset = 0;
    const limit = 1000; // Large batches, metadata only = fast

    logger.info('CHROMA_SYNC', 'Fetching existing Chroma document IDs...', { project: targetProject });

    while (true) {
      const result = await chromaMcp.callTool('chroma_get_documents', {
        collection_name: this.collectionName,
        limit: limit,
        offset: offset,
        where: { project: targetProject },
        include: ['metadatas']
      }) as any;

      // chroma_get_documents returns flat arrays: { ids, metadatas, documents }
      const metadatas = result?.metadatas || [];

        const parsed = JSON.parse(data.text);
        const metadatas = parsed.metadatas || [];

        if (metadatas.length === 0) {
          break; // No more documents
        }

        // Extract SQLite IDs from metadata
        for (const meta of metadatas) {
          if (meta.sqlite_id) {
            if (meta.doc_type === 'observation') {
              observationIds.add(meta.sqlite_id);
            } else if (meta.doc_type === 'session_summary') {
              summaryIds.add(meta.sqlite_id);
            } else if (meta.doc_type === 'user_prompt') {
              promptIds.add(meta.sqlite_id);
            }
          }
          if (meta.doc_type === 'thought' && meta.thought_id) {
            thoughtIds.add(meta.thought_id);
          }
        }

        offset += limit;

        logger.debug('CHROMA_SYNC', 'Fetched batch of existing IDs', {
          project: this.project,
          offset,
          batchSize: metadatas.length
        });
      } catch (error) {
        logger.error('CHROMA_SYNC', 'Failed to fetch existing IDs', { project: this.project }, error as Error);
        throw error;
      }

      // Extract SQLite IDs from metadata
      for (const meta of metadatas) {
        if (meta && meta.sqlite_id) {
          const sqliteId = meta.sqlite_id as number;
          if (meta.doc_type === 'observation') {
            observationIds.add(sqliteId);
          } else if (meta.doc_type === 'session_summary') {
            summaryIds.add(sqliteId);
          } else if (meta.doc_type === 'user_prompt') {
            promptIds.add(sqliteId);
          }
        }
      }

      offset += limit;

      logger.debug('CHROMA_SYNC', 'Fetched batch of existing IDs', {
        project: targetProject,
        offset,
        batchSize: metadatas.length
      });
    }

    logger.info('CHROMA_SYNC', 'Existing IDs fetched', {
      project: targetProject,
      observations: observationIds.size,
      summaries: summaryIds.size,
      prompts: promptIds.size,
      thoughts: thoughtIds.size
    });

    return { observations: observationIds, summaries: summaryIds, prompts: promptIds, thoughts: thoughtIds };
  }

  /**
   * Backfill: Sync all observations missing from Chroma
   * Reads from SQLite in paginated batches to avoid OOM on large datasets
   * Throws error if backfill fails
   */
  async ensureBackfilled(projectOverride?: string): Promise<void> {
    const backfillProject = projectOverride ?? this.project;
    logger.info('CHROMA_SYNC', 'Starting smart backfill', { project: backfillProject });

    await this.ensureCollectionExists();

    // Fetch existing IDs from Chroma (fast, metadata only)
    const existing = await this.getExistingChromaIds(backfillProject);

    const db = new SessionStore();
    const PAGE_SIZE = 500; // Process 500 rows at a time to limit memory usage

    try {
      // Build exclusion list for observations
      // Filter to validated positive integers before interpolating into SQL
      const existingObsIds = Array.from(existing.observations).filter(id => Number.isInteger(id) && id > 0);
      const obsExclusionClause = existingObsIds.length > 0
        ? `AND id NOT IN (${existingObsIds.join(',')})`
        : '';

      const totalObsCount = db.db.prepare(`
        SELECT COUNT(*) as count FROM observations WHERE project = ?
      `).get(backfillProject) as { count: number };

      // Paginated observation backfill
      let obsOffset = 0;
      let totalObsDocs = 0;
      let totalObsRows = 0;

      while (true) {
        const observations = db.db.prepare(`
          SELECT * FROM observations
          WHERE project = ? ${obsExclusionClause}
          ORDER BY id ASC
          LIMIT ? OFFSET ?
        `).all(this.project, PAGE_SIZE, obsOffset) as StoredObservation[];

        if (observations.length === 0) break;
        totalObsRows += observations.length;

        // Format observation documents for this page
        const pageDocs: ChromaDocument[] = [];
        for (const obs of observations) {
          pageDocs.push(...this.formatObservationDocs(obs));
        }

        // Sync page in batches
        for (let i = 0; i < pageDocs.length; i += this.BATCH_SIZE) {
          const batch = pageDocs.slice(i, i + this.BATCH_SIZE);
          await this.addDocuments(batch);
        }

        totalObsDocs += pageDocs.length;
        obsOffset += PAGE_SIZE;

        logger.debug('CHROMA_SYNC', 'Observation backfill progress', {
          project: this.project,
          rowsProcessed: totalObsRows,
          docsCreated: totalObsDocs
        });
      }

      logger.info('CHROMA_SYNC', 'Backfilled observations', {
        project: this.project,
        missing: totalObsRows,
        existing: existing.observations.size,
        total: totalObsCount.count
      });

      // Build exclusion list for summaries
      const existingSummaryIds = Array.from(existing.summaries).filter(id => Number.isInteger(id) && id > 0);
      const summaryExclusionClause = existingSummaryIds.length > 0
        ? `AND id NOT IN (${existingSummaryIds.join(',')})`
        : '';

      const totalSummaryCount = db.db.prepare(`
        SELECT COUNT(*) as count FROM session_summaries WHERE project = ?
      `).get(backfillProject) as { count: number };

      // Paginated summary backfill
      let summaryOffset = 0;
      let totalSummaryDocs = 0;
      let totalSummaryRows = 0;

      while (true) {
        const summaries = db.db.prepare(`
          SELECT * FROM session_summaries
          WHERE project = ? ${summaryExclusionClause}
          ORDER BY id ASC
          LIMIT ? OFFSET ?
        `).all(this.project, PAGE_SIZE, summaryOffset) as StoredSummary[];

        if (summaries.length === 0) break;
        totalSummaryRows += summaries.length;

        const pageDocs: ChromaDocument[] = [];
        for (const summary of summaries) {
          pageDocs.push(...this.formatSummaryDocs(summary));
        }

        for (let i = 0; i < pageDocs.length; i += this.BATCH_SIZE) {
          const batch = pageDocs.slice(i, i + this.BATCH_SIZE);
          await this.addDocuments(batch);
        }

        totalSummaryDocs += pageDocs.length;
        summaryOffset += PAGE_SIZE;
      }

      logger.info('CHROMA_SYNC', 'Backfilled summaries', {
        project: this.project,
        missing: totalSummaryRows,
        existing: existing.summaries.size,
        total: totalSummaryCount.count
      });

      // Build exclusion list for prompts
      const existingPromptIds = Array.from(existing.prompts).filter(id => Number.isInteger(id) && id > 0);
      const promptExclusionClause = existingPromptIds.length > 0
        ? `AND up.id NOT IN (${existingPromptIds.join(',')})`
        : '';

      const totalPromptCount = db.db.prepare(`
        SELECT COUNT(*) as count
        FROM user_prompts up
        JOIN sdk_sessions s ON up.content_session_id = s.content_session_id
        WHERE s.project = ?
      `).get(backfillProject) as { count: number };

      // Paginated prompt backfill
      let promptOffset = 0;
      let totalPromptDocs = 0;
      let totalPromptRows = 0;

      while (true) {
        const prompts = db.db.prepare(`
          SELECT
            up.*,
            s.project,
            s.memory_session_id
          FROM user_prompts up
          JOIN sdk_sessions s ON up.content_session_id = s.content_session_id
          WHERE s.project = ? ${promptExclusionClause}
          ORDER BY up.id ASC
          LIMIT ? OFFSET ?
        `).all(this.project, PAGE_SIZE, promptOffset) as StoredUserPrompt[];

        if (prompts.length === 0) break;
        totalPromptRows += prompts.length;

        const pageDocs: ChromaDocument[] = [];
        for (const prompt of prompts) {
          pageDocs.push(this.formatUserPromptDoc(prompt));
        }

        for (let i = 0; i < pageDocs.length; i += this.BATCH_SIZE) {
          const batch = pageDocs.slice(i, i + this.BATCH_SIZE);
          await this.addDocuments(batch);
        }

        totalPromptDocs += pageDocs.length;
        promptOffset += PAGE_SIZE;
      }

      logger.info('CHROMA_SYNC', 'Backfilled user prompts', {
        project: this.project,
        missing: totalPromptRows,
        existing: existing.prompts.size,
        total: totalPromptCount.count
      });

      // Format all prompt documents
      const promptDocs: ChromaDocument[] = [];
      for (const prompt of prompts) {
        promptDocs.push(this.formatUserPromptDoc(prompt));
      }

      // Sync in batches
      for (let i = 0; i < promptDocs.length; i += this.BATCH_SIZE) {
        const batch = promptDocs.slice(i, i + this.BATCH_SIZE);
        await this.addDocuments(batch);

        logger.debug('CHROMA_SYNC', 'Backfill progress', {
          project: this.project,
          progress: `${Math.min(i + this.BATCH_SIZE, promptDocs.length)}/${promptDocs.length}`
        });
      }

      // Get all thoughts for this project
      const allThoughts = db.getThoughts(this.project);

      // Filter out thoughts already in Chroma
      const missingThoughts = allThoughts.filter(t => !existing.thoughts.has(t.id));

      logger.info('CHROMA_SYNC', 'Backfilling thoughts', {
        project: this.project,
        missing: missingThoughts.length,
        existing: existing.thoughts.size,
        total: allThoughts.length
      });

      // Format thought documents
      const thoughtDocs: ChromaDocument[] = missingThoughts.map(thought => ({
        id: `thought_${thought.id}`,
        document: thought.thinking_text,
        metadata: {
          doc_type: 'thought',
          thought_id: thought.id,
          memory_session_id: thought.memory_session_id,
          project: thought.project,
          message_index: thought.message_index ?? 0,
          created_at_epoch: thought.created_at_epoch
        }
      }));

      // Sync in batches
      for (let i = 0; i < thoughtDocs.length; i += this.BATCH_SIZE) {
        const batch = thoughtDocs.slice(i, i + this.BATCH_SIZE);
        await this.addDocuments(batch);

        logger.debug('CHROMA_SYNC', 'Backfill progress', {
          project: this.project,
          progress: `${Math.min(i + this.BATCH_SIZE, thoughtDocs.length)}/${thoughtDocs.length}`
        });
      }

      logger.info('CHROMA_SYNC', 'Smart backfill complete', {
        project: backfillProject,
        synced: {
          observationDocs: allDocs.length,
          summaryDocs: summaryDocs.length,
          promptDocs: promptDocs.length,
          thoughtDocs: thoughtDocs.length
        },
        skipped: {
          observations: existing.observations.size,
          summaries: existing.summaries.size,
          prompts: existing.prompts.size,
          thoughts: existing.thoughts.size
        }
      });

    } catch (error) {
      logger.error('CHROMA_SYNC', 'Backfill failed', { project: backfillProject }, error as Error);
      throw new Error(`Backfill failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      db.close();
    }
  }

  /**
   * Query Chroma collection for semantic search via MCP
   * Used by SearchManager for vector-based search
   */
  async queryChroma(
    query: string,
    limit: number,
    whereFilter?: Record<string, any>
  ): Promise<{ ids: number[]; distances: number[]; metadatas: any[] }> {
    await this.ensureCollectionExists();

    try {
      const chromaMcp = ChromaMcpManager.getInstance();
      const results = await chromaMcp.callTool('chroma_query_documents', {
        collection_name: this.collectionName,
        query_texts: [query],
        n_results: limit,
        ...(whereFilter && { where: whereFilter }),
        include: ['documents', 'metadatas', 'distances']
      }) as any;

      // chroma_query_documents returns nested arrays (one per query text)
      // We always pass a single query text, so we access [0]
      const ids: number[] = [];
      const seen = new Set<number>();
      const docIds = results?.ids?.[0] || [];
      const rawMetadatas = results?.metadatas?.[0] || [];
      const rawDistances = results?.distances?.[0] || [];

      // Build deduplicated arrays that stay index-aligned:
      // Multiple Chroma docs map to the same SQLite ID (one per field).
      // Keep the first (best-ranked) distance and metadata per SQLite ID.
      const metadatas: any[] = [];
      const distances: number[] = [];

      for (let i = 0; i < docIds.length; i++) {
        const docId = docIds[i];
        // Extract sqlite_id from document ID (supports three formats):
        // - obs_{id}_narrative, obs_{id}_fact_0, etc (observations)
        // - summary_{id}_request, summary_{id}_learned, etc (session summaries)
        // - prompt_{id} (user prompts)
        const obsMatch = docId.match(/obs_(\d+)_/);
        const summaryMatch = docId.match(/summary_(\d+)_/);
        const promptMatch = docId.match(/prompt_(\d+)/);

        let sqliteId: number | null = null;
        if (obsMatch) {
          sqliteId = parseInt(obsMatch[1], 10);
        } else if (summaryMatch) {
          sqliteId = parseInt(summaryMatch[1], 10);
        } else if (promptMatch) {
          sqliteId = parseInt(promptMatch[1], 10);
        }

        if (sqliteId !== null && !seen.has(sqliteId)) {
          seen.add(sqliteId);
          ids.push(sqliteId);
          metadatas.push(rawMetadatas[i] ?? null);
          distances.push(rawDistances[i] ?? 0);
        }
      }

      return { ids, distances, metadatas };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Check for connection errors
      const isConnectionError =
        errorMessage.includes('ECONNREFUSED') ||
        errorMessage.includes('ENOTFOUND') ||
        errorMessage.includes('fetch failed') ||
        errorMessage.includes('subprocess closed') ||
        errorMessage.includes('timed out');

      if (isConnectionError) {
        await this.safeResetConnection();
        logger.error('CHROMA_SYNC', 'Connection lost during query',
          { project: this.project, query }, error as Error);
        throw new Error(`Chroma query failed - connection lost: ${errorMessage}`);
      }

      logger.error('CHROMA_SYNC', 'Query failed', { project: this.project, query }, error as Error);
      throw error;
    }
  }

  /**
   * Backfill all projects that have observations in SQLite but may be missing from Chroma.
   * Uses a single shared ChromaSync('claude-mem') instance and Chroma connection.
   * Per-project scoping is passed as a parameter to ensureBackfilled(), avoiding
   * instance state mutation. All documents land in the cm__claude-mem collection
   * with project scoped via metadata, matching how DatabaseManager and SearchManager operate.
   * Designed to be called fire-and-forget on worker startup.
   */
  static async backfillAllProjects(): Promise<void> {
    const db = new SessionStore();
    const sync = new ChromaSync('claude-mem');
    try {
      const projects = db.db.prepare(
        'SELECT DISTINCT project FROM observations WHERE project IS NOT NULL AND project != ?'
      ).all('') as { project: string }[];

      logger.info('CHROMA_SYNC', `Backfill check for ${projects.length} projects`);

      for (const { project } of projects) {
        try {
          await sync.ensureBackfilled(project);
        } catch (error) {
          logger.error('CHROMA_SYNC', `Backfill failed for project: ${project}`, {}, error as Error);
          // Continue to next project — don't let one failure stop others
        }
      }
    } finally {
      await sync.close();
      db.close();
    }
  }

  /**
   * Close the ChromaSync instance
   * ChromaMcpManager is a singleton and manages its own lifecycle
   * We don't close it here - it's closed during graceful shutdown
   */
  /**
   * Safely reset connection state — closes transport BEFORE nulling reference.
   * Prevents orphaned chroma-mcp subprocesses.
   */
  private async safeResetConnection(): Promise<void> {
    const t = this.transport;
    const c = this.client;
    this.connected = false;
    this.client = null;
    this.transport = null;
    if (c) { try { await c.close(); } catch {} }
    if (t) { try { await t.close(); } catch {} }
  }

  async close(): Promise<void> {
    await this.safeResetConnection();
    this.connectionPromise = null;
    logger.info('CHROMA_SYNC', 'Chroma client and subprocess closed', { project: this.project });
  }
}
