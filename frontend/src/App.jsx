import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import {
  Bot,
  MessageSquare,
  Files,
  Settings,
  Plus,
  Search,
  Upload,
  Send,
  Sparkles,
  FileText,
  ChevronDown,
  PanelLeftClose,
  Database,
  ShieldCheck,
  LoaderCircle,
  CheckCircle,
  AlertCircle,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function readApiResponse(response, fallbackMessage) {
  const responseText = await response.text();
  let data = {};

  if (responseText) {
    try {
      data = JSON.parse(responseText);
    } catch {
      data = { detail: responseText.trim() };
    }
  }

  if (!response.ok) {
    const detail =
      typeof data.detail === "string" && data.detail.trim()
        ? data.detail.trim()
        : fallbackMessage;

    throw new Error(detail);
  }

  return data;
}

function isNetworkError(error) {
  return error instanceof TypeError;
}

function getRequestErrorMessage(error, fallbackMessage) {
  if (isNetworkError(error)) {
    return "Could not reach the backend. Check that NEXUS AI is running and try again.";
  }

  return error.message || fallbackMessage;
}

function mapDocuments(data) {
  if (!Array.isArray(data.documents)) {
    throw new Error("The backend returned an invalid document list.");
  }

  return data.documents.map((document) => ({
    id: document.document_id,
    name: document.document_name,
    type: "PDF",
    chunks: document.chunks,
  }));
}

function App() {
  const [activePage, setActivePage] = useState("Chat");
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);

  const [search, setSearch] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState([]);
  const [isAsking, setIsAsking] = useState(false);
  const [chatError, setChatError] = useState("");

  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const fileInputRef = useRef(null);

  const [apiStatus, setApiStatus] = useState("Checking API");
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState("");

  const refreshDocuments = useCallback(async () => {
    const response = await fetch(`${API_URL}/documents`);

    const data = await readApiResponse(
      response,
      "Could not retrieve documents."
    );
    const loadedDocuments = mapDocuments(data);

    setDocuments(loadedDocuments);

    return loadedDocuments;
  }, []);

  useEffect(() => {
    async function loadDocuments() {
      try {
        setApiStatus("Checking API");

        const loadedDocuments = await refreshDocuments();

        setSelectedDocument((current) => {
          if (current) {
            const stillExists = loadedDocuments.find(
              (document) => document.id === current.id
            );

            if (stillExists) return stillExists;
          }

          return loadedDocuments[0] ?? null;
        });

        setApiStatus("Connected");
        setDocumentsError("");
      } catch (error) {
        console.error("Document loading error:", error);
        setApiStatus("Disconnected");
        setDocumentsError(
          "Could not connect to the backend. Make sure FastAPI is running."
        );
      } finally {
        setDocumentsLoading(false);
      }
    }

    loadDocuments();
  }, [refreshDocuments]);

  const filteredDocuments = documents.filter((document) =>
    document.name.toLowerCase().includes(search.toLowerCase())
  );

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (file) {
      setSelectedFile(file);
      setUploadStatus(null);
    }
  }

  async function handleUpload() {
    if (!selectedFile || isUploading) return;

    if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus({
        type: "error",
        message: "Only PDF files are supported.",
      });
      return;
    }

    setIsUploading(true);
    setUploadStatus(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await readApiResponse(
        response,
        "Document upload failed."
      );

      const loadedDocuments = await refreshDocuments();

      const uploadedDocument = loadedDocuments.find(
        (document) => document.id === data.document_id
      );

      if (uploadedDocument) {
        setSelectedDocument(uploadedDocument);
        setMessages([]);
        setChatError("");
      }

      setUploadStatus({
        type: "success",
        message: `${data.document_name} uploaded successfully. ${data.chunks_stored} chunks stored.`,
      });

      setSelectedFile(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }

      setApiStatus("Connected");
    } catch (error) {
      console.error("Upload error:", error);

      if (isNetworkError(error)) {
        setApiStatus("Disconnected");
      }

      setUploadStatus({
        type: "error",
        message: getRequestErrorMessage(
          error,
          "Could not upload the document."
        ),
      });
    } finally {
      setIsUploading(false);
    }
  }

  function handleDocumentSelect(document) {
    setSelectedDocument(document);
    setMessages([]);
    setChatError("");
    setActivePage("Chat");
  }

  async function handleAsk() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || !selectedDocument || isAsking) {
      return;
    }

    setIsAsking(true);
    setChatError("");
    setQuestion("");

    try {
      const response = await fetch(
        `${API_URL}/documents/${selectedDocument.id}/ask`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: trimmedQuestion,
          }),
        }
      );

      const data = await readApiResponse(
        response,
        "The request failed."
      );

      const fullAnswer = data.answer || "";
      const sourceMarker = "SOURCES:";
      const sourceIndex = fullAnswer.indexOf(sourceMarker);

      const answerText =
        sourceIndex >= 0
          ? fullAnswer.slice(0, sourceIndex).trim()
          : fullAnswer.trim();

      const sourcesText =
        sourceIndex >= 0
          ? fullAnswer
              .slice(sourceIndex + sourceMarker.length)
              .trim()
              .split("\n")
              .map((source) => source.replace(/^- /, "").trim())
              .filter(Boolean)
          : [];

      setMessages((current) => [
        ...current,
        {
          question: trimmedQuestion,
          answer: answerText,
          sources: sourcesText,
          documentName: selectedDocument.name,
        },
      ]);
    } catch (error) {
      console.error("Question error:", error);

      if (isNetworkError(error)) {
        setApiStatus("Disconnected");
      }

      setChatError(
        getRequestErrorMessage(
          error,
          "Could not get an answer from NEXUS AI."
        )
      );
      setQuestion(trimmedQuestion);
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Bot size={23} />
          </div>

          <div>
            <h1>NEXUS AI</h1>
            <span>ENTERPRISE AI</span>
          </div>

          <button className="icon-button sidebar-toggle" title="Sidebar">
            <PanelLeftClose size={17} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          <button
            className={`nav-item ${activePage === "Chat" ? "active" : ""}`}
            onClick={() => setActivePage("Chat")}
          >
            <MessageSquare size={18} />
            <span>AI Assistant</span>
          </button>

          <button
            className={`nav-item ${activePage === "Documents" ? "active" : ""}`}
            onClick={() => setActivePage("Documents")}
          >
            <Files size={18} />
            <span>Documents</span>
            <span className="nav-count">{documents.length}</span>
          </button>

          <button
            className={`nav-item ${activePage === "Settings" ? "active" : ""}`}
            onClick={() => setActivePage("Settings")}
          >
            <Settings size={18} />
            <span>Settings</span>
          </button>
        </nav>

        <div className="sidebar-divider" />

        <div className="documents-heading">
          <span>YOUR DOCUMENTS</span>
          <span>{documents.length}</span>
        </div>

        <div className="document-search">
          <Search size={15} />
          <input
            type="text"
            placeholder="Search documents..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>

        <div className="document-list">
          {documentsLoading ? (
            <span className="document-meta">Loading documents...</span>
          ) : documentsError ? (
            <span className="document-meta">Documents unavailable</span>
          ) : filteredDocuments.length === 0 ? (
            <span className="document-meta">No documents found</span>
          ) : (
            filteredDocuments.map((document) => (
              <button
                key={document.id}
                className={`document-item ${
                  selectedDocument?.id === document.id ? "selected" : ""
                }`}
                onClick={() => handleDocumentSelect(document)}
              >
                <div className="document-icon">
                  <FileText size={17} />
                </div>

                <div className="document-info">
                  <span className="document-name">{document.name}</span>
                  <span className="document-meta">
                    {document.chunks} chunks
                  </span>
                </div>
              </button>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="status-dot" />
            <span>System operational</span>
          </div>

          <div className="profile">
            <div className="avatar">M</div>
            <div className="profile-info">
              <strong>Workspace Admin</strong>
              <span>Administrator</span>
            </div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            <span>Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>{activePage}</strong>
          </div>

          <div className="topbar-right">
            <div className="topbar-status">
              <span
                className="status-dot"
                style={{
                  background:
                    apiStatus === "Connected"
                      ? "#25b77a"
                      : apiStatus === "Disconnected"
                        ? "#dc6262"
                        : "#e2a83e",
                }}
              />
              {apiStatus === "Connected"
                ? "API Connected"
                : apiStatus === "Disconnected"
                  ? "API Disconnected"
                  : "Checking API"}
            </div>

            <div className="topbar-divider" />

            <button className="icon-button" title="Settings">
              <Settings size={18} />
            </button>
          </div>
        </header>

        {activePage === "Chat" && (
          <section className="chat-workspace">
            <div className="chat-header">
              <div>
                <div className="eyebrow">
                  <Sparkles size={14} />
                  INTELLIGENT DOCUMENT SEARCH
                </div>
                <h2>AI Assistant</h2>
                <p>Ask questions and get answers from your documents.</p>
              </div>

              <button className="selected-document">
                <FileText size={17} />
                <span>
                  {selectedDocument?.name ?? "Select a document"}
                </span>
                <ChevronDown size={15} />
              </button>
            </div>

            <div className="chat-content">
              {messages.length === 0 && !isAsking ? (
                <div className="welcome-content">
                  <div className="welcome-icon">
                    <Bot size={30} />
                    <span>
                      <Sparkles size={13} />
                    </span>
                  </div>

                  <h2>Your documents, understood.</h2>
                  <p>
                    Ask questions about your uploaded documents.
                    NEXUS AI retrieves relevant information and generates
                    grounded answers with sources.
                  </p>

                  {documentsError && <p>{documentsError}</p>}

                  <div className="feature-cards">
                    <div className="feature-card">
                      <div className="feature-icon purple">
                        <Database size={19} />
                      </div>
                      <strong>Hybrid Retrieval</strong>
                      <span>Semantic search and keyword matching.</span>
                    </div>

                    <div className="feature-card">
                      <div className="feature-icon blue">
                        <ShieldCheck size={19} />
                      </div>
                      <strong>Grounded Answers</strong>
                      <span>Answers based on your selected document.</span>
                    </div>

                    <div className="feature-card">
                      <div className="feature-icon green">
                        <FileText size={19} />
                      </div>
                      <strong>Source References</strong>
                      <span>Trace answers back to document sections.</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div
                  style={{
                    width: "100%",
                    maxWidth: "850px",
                    margin: "0 auto",
                    padding: "24px 10px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "24px",
                  }}
                >
                  {messages.map((message, index) => (
                    <div
                      key={`${index}-${message.question}`}
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "14px",
                      }}
                    >
                      <div
                        style={{
                          alignSelf: "flex-end",
                          maxWidth: "85%",
                          background: "#635bdb",
                          color: "#ffffff",
                          padding: "14px 18px",
                          borderRadius: "16px 16px 4px 16px",
                          lineHeight: 1.6,
                          overflowWrap: "anywhere",
                        }}
                      >
                        {message.question}
                      </div>

                      <div
                        style={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: "12px",
                        }}
                      >
                        <div
                          style={{
                            background: "#eeecff",
                            color: "#635bdb",
                            borderRadius: "12px",
                            padding: "9px",
                            flexShrink: 0,
                          }}
                        >
                          <Bot size={20} />
                        </div>

                        <div style={{ minWidth: 0, flex: 1 }}>
                          <div
                            style={{
                              fontWeight: 700,
                              marginBottom: "8px",
                              color: "#25283a",
                            }}
                          >
                            NEXUS AI
                          </div>

                          <div
                            style={{
                              color: "#45495b",
                              lineHeight: 1.8,
                              whiteSpace: "pre-wrap",
                              overflowWrap: "anywhere",
                            }}
                          >
                            {message.answer}
                          </div>

                          {message.sources.length > 0 && (
                            <div
                              style={{
                                marginTop: "18px",
                                padding: "14px",
                                background: "#f5f5fa",
                                border: "1px solid #e8e8f0",
                                borderRadius: "10px",
                              }}
                            >
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "7px",
                                  fontWeight: 700,
                                  fontSize: "13px",
                                  marginBottom: "9px",
                                  color: "#45495b",
                                }}
                              >
                                <FileText size={15} />
                                Sources
                              </div>

                              {message.sources.map((source, sourceIndex) => (
                                <div
                                  key={`${sourceIndex}-${source}`}
                                  style={{
                                    fontSize: "13px",
                                    color: "#65697a",
                                    padding: "4px 0",
                                  }}
                                >
                                  {source}
                                </div>
                              ))}
                            </div>
                          )}

                          <div
                            style={{
                              fontSize: "12px",
                              color: "#9295a3",
                              marginTop: "10px",
                            }}
                          >
                            Document: {message.documentName}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}

                  {isAsking && (
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "10px",
                        color: "#77798a",
                        padding: "10px",
                      }}
                    >
                      <LoaderCircle size={19} className="spin-animation" />
                      NEXUS AI is searching your document and generating an
                      answer...
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="composer-area">
              {chatError && (
                <p
                  style={{
                    color: "#c03939",
                    fontSize: "13px",
                    marginBottom: "10px",
                  }}
                >
                  {chatError}
                </p>
              )}

              <div className="composer">
                <textarea
                  placeholder="Ask a question about your document..."
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleAsk();
                    }
                  }}
                  rows={2}
                  disabled={isAsking}
                />

                <div className="composer-footer">
                  <span>
                    <ShieldCheck size={14} />
                    Responses are grounded in your documents
                  </span>

                  <button
                    className="send-button"
                    onClick={handleAsk}
                    disabled={
                      !question.trim() || !selectedDocument || isAsking
                    }
                    title="Send question"
                  >
                    {isAsking ? (
                      <LoaderCircle size={16} className="spin-animation" />
                    ) : (
                      <Send size={16} />
                    )}
                  </button>
                </div>
              </div>

              <p className="disclaimer">
                NEXUS AI can make mistakes. Verify important information
                against the original document.
              </p>
            </div>
          </section>
        )}

        {activePage === "Documents" && (
          <section className="page-content">
            <div className="page-heading">
              <div>
                <div className="eyebrow">KNOWLEDGE BASE</div>
                <h2>Documents</h2>
                <p>Manage the documents available to your AI assistant.</p>
              </div>
            </div>

            <div className="upload-card">
              <div className="upload-icon">
                <Upload size={25} />
              </div>

              <h3>Add a document</h3>
              <p>Select a PDF to add it to your knowledge base.</p>

              <label className="upload-button">
                <Plus size={17} />
                Choose PDF
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={handleFileChange}
                  hidden
                  disabled={isUploading}
                />
              </label>

              {selectedFile && (
                <div className="selected-file">
                  <FileText size={16} />
                  <span>{selectedFile.name}</span>
                </div>
              )}

              {selectedFile && (
                <button
                  className="upload-button"
                  onClick={handleUpload}
                  disabled={isUploading}
                  style={{
                    marginTop: "12px",
                    opacity: isUploading ? 0.65 : 1,
                    cursor: isUploading ? "wait" : "pointer",
                  }}
                >
                  {isUploading ? (
                    <>
                      <LoaderCircle size={16} className="spin-animation" />
                      Uploading and processing...
                    </>
                  ) : (
                    <>
                      <Upload size={16} />
                      Upload PDF
                    </>
                  )}
                </button>
              )}

              {uploadStatus && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "8px",
                    marginTop: "16px",
                    padding: "12px",
                    borderRadius: "9px",
                    background:
                      uploadStatus.type === "success" ? "#eaf8f0" : "#fff0f0",
                    color:
                      uploadStatus.type === "success" ? "#21764b" : "#b83232",
                    fontSize: "13px",
                    lineHeight: 1.5,
                    textAlign: "left",
                  }}
                >
                  {uploadStatus.type === "success" ? (
                    <CheckCircle size={17} />
                  ) : (
                    <AlertCircle size={17} />
                  )}
                  <span>{uploadStatus.message}</span>
                </div>
              )}
            </div>

            <div className="document-table">
              <div className="table-heading">
                <h3>Uploaded documents</h3>
                <span>
                  {documentsLoading
                    ? "Loading..."
                    : `${documents.length} documents`}
                </span>
              </div>

              {documentsError && <p>{documentsError}</p>}

              {documents.map((document) => (
                <div className="table-row" key={document.id}>
                  <FileText size={19} />
                  <span>{document.name}</span>
                  <span>{document.chunks} chunks</span>
                  <span className="file-type">{document.type}</span>
                </div>
              ))}
            </div>
          </section>
        )}

        {activePage === "Settings" && (
          <section className="page-content">
            <div className="eyebrow">PREFERENCES</div>
            <h2>Settings</h2>
            <p>Application settings will be configured here.</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
